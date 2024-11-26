import argparse
import sys
import time
import cv2
import numpy as np
import math
import random
import string
import os
import glob
from tf.transformations import quaternion_from_euler
from google.protobuf import wrappers_pb2
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import graph_nav_util
from bosdyn.client.exceptions import ResponseError
from bosdyn.client import math_helpers
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.power import PowerClient
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import gripper_camera_param_pb2, header_pb2
from bosdyn.client.gripper_camera_param import GripperCameraParamClient

from pallet_detection_stream import PalletDetectionLive
from read_barcodes import BarcodeScanner

PICTURE_COUNT = 3
TRAJECTORY_DURATION = 5 # [sec]
DISTANCE_TO_OBJECT = 1.0 # [m]
INCREASE_TRAJECTORY_DISTANCE = -0.25
TRAJECTORY_FRAME = VISION_FRAME_NAME # or ODOM_FRAME_NAME

class WalkToPallet:
    def __init__(self, config) -> None:
        self.config = config
        bosdyn.client.util.setup_logging(config.verbose)

        self.sdk = bosdyn.client.create_standard_sdk('walk_to_pallet')
        self.robot = self.sdk.create_robot(config.hostname)
        self.command_client = None
        self.manipulation_api_client = None
        self.image_client = None

        self.image_folder = "/img"
        self.map_folder = "/map/downloaded_graph/"

        # Waypoints
        self.pallet_1 = 'aided-oxen-M9nz6XZIwQEHvBuxbjqqCg=='
        self.pallet_2 = 'td'
        self.pallet_3 = 'afeard-cowrie-IQI2zULHrcv.f9ByjxA1Bg=='
        self.home = 'miffed-grub-Um30vzkWexCmgCVEeZyKiw=='
        

    def _setup_robot(self):
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        assert self.robot.has_arm(), 'Robot requires an arm to run this example.'

        assert not self.robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.robot_command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self.gripper_camera_param_client = self.robot.ensure_client(GripperCameraParamClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.power_client = self.robot.ensure_client(PowerClient.default_service_name)

        # Filepath for uploading a saved graph's and snapshots too.
        if self.map_folder[-1] == '/':
            self._upload_filepath = self.map_folder[:-1]
        else:
            self._upload_filepath = self.map_folder


        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

    def set_gripper_camera_params(self, options):
        # Camera resolution
        camera_mode = None
        if options.resolution is not None:
            if options.resolution == '640x480':
                camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480
            elif options.resolution == '1280x720':
                camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1280_720
            elif options.resolution == '1920x1080':
                camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1920_1080
            elif options.resolution == '3840x2160':
                camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_3840_2160
            elif options.resolution == '4096x2160':
                camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4096_2160
            elif options.resolution == '4208x3120':
                camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4208_3120

        # Other settings
        brightness = None
        if options.brightness is not None:
            brightness = wrappers_pb2.FloatValue(value=options.brightness)

        contrast = None
        if options.contrast is not None:
            contrast = wrappers_pb2.FloatValue(value=options.contrast)

        saturation = None
        if options.saturation is not None:
            saturation = wrappers_pb2.FloatValue(value=options.saturation)

        gain = None
        if options.gain is not None:
            gain = wrappers_pb2.FloatValue(value=options.gain)

        if options.manual_focus is not None and options.auto_focus and options.auto_focus == 'on':
            print('Error: cannot specify both a manual focus value and enable auto-focus.')
            sys.exit(1)

        manual_focus = None
        auto_focus = None
        if options.manual_focus:
            manual_focus = wrappers_pb2.FloatValue(value=options.manual_focus)
            auto_focus = wrappers_pb2.BoolValue(value=False)

        if options.auto_focus is not None:
            auto_focus_enabled = options.auto_focus == 'on'
            auto_focus = wrappers_pb2.BoolValue(value=auto_focus_enabled)

        if options.exposure is not None and options.auto_exposure and options.auto_exposure == 'on':
            print('Error: cannot specify both a manual exposure value and enable auto-exposure.')
            sys.exit(1)

        exposure = None
        auto_exposure = None
        if options.exposure is not None:
            exposure = wrappers_pb2.FloatValue(value=options.exposure)
            auto_exposure = wrappers_pb2.BoolValue(value=False)

        if options.auto_exposure:
            auto_exposure_enabled = options.auto_exposure == 'on'
            auto_exposure = wrappers_pb2.BoolValue(value=auto_exposure_enabled)

        hdr = None
        if options.hdr_mode is not None:
            if options.hdr_mode == 'off':
                hdr = gripper_camera_param_pb2.HDR_OFF
            elif options.hdr_mode == 'auto':
                hdr = gripper_camera_param_pb2.HDR_AUTO
            elif options.hdr_mode == 'manual1':
                hdr = gripper_camera_param_pb2.HDR_MANUAL_1
            elif options.hdr_mode == 'manual2':
                hdr = gripper_camera_param_pb2.HDR_MANUAL_2
            elif options.hdr_mode == 'manual3':
                hdr = gripper_camera_param_pb2.HDR_MANUAL_3
            elif options.hdr_mode == 'manual4':
                hdr = gripper_camera_param_pb2.HDR_MANUAL_4

        led_mode = None
        if options.led_mode is not None:
            if options.led_mode == 'off':
                led_mode = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_OFF
            elif options.led_mode == 'torch':
                led_mode = gripper_camera_param_pb2.GripperCameraParams.LED_MODE_TORCH

        led_torch_brightness = None
        if options.led_torch_brightness is not None:
            led_torch_brightness = wrappers_pb2.FloatValue(value=options.led_torch_brightness)


        gamma = None
        if options.gamma is not None:
            gamma = wrappers_pb2.FloatValue(value=options.gamma)

        sharpness = None
        if options.sharpness is not None:
            sharpness = wrappers_pb2.FloatValue(value=options.sharpness)

        if options.white_balance_temperature is not None and options.white_balance_temperature_auto and options.white_balance_temperature_auto == 'on':
            print(
                'Error: cannot specify both a manual white_balance_temperature value and enable white_balance_temperature_auto.'
            )
            sys.exit(1)

        white_balance_temperature = None
        white_balance_temperature_auto = None
        if options.white_balance_temperature is not None:
            white_balance_temperature = wrappers_pb2.FloatValue(value=options.white_balance_temperature)
            white_balance_temperature_auto = wrappers_pb2.BoolValue(value=False)

        if options.white_balance_temperature_auto:
            white_balance_temperature_auto_enabled = options.white_balance_temperature_auto == 'on'
            white_balance_temperature_auto = wrappers_pb2.BoolValue(
                value=white_balance_temperature_auto_enabled)
            
        # Construct the GripperCameraParams
        params = gripper_camera_param_pb2.GripperCameraParams(
            camera_mode=camera_mode, brightness=brightness, contrast=contrast, gain=gain,
            saturation=saturation, focus_absolute=manual_focus, focus_auto=auto_focus,
            exposure_absolute=exposure, exposure_auto=auto_exposure, hdr=hdr, led_mode=led_mode,
            led_torch_brightness=led_torch_brightness, gamma=gamma, sharpness=sharpness,
            white_balance_temperature=white_balance_temperature,
            white_balance_temperature_auto=white_balance_temperature_auto)
        
        request = gripper_camera_param_pb2.GripperCameraParamRequest(params=params)
        # Send the request
        response = self.gripper_camera_param_client.set_camera_params(request)
        print('Sent request')

        if response.header.error and response.header.error.code != header_pb2.CommonError.CODE_OK:
            print('Got an error:')
            print(response.header.error)

        print('Now querying robot for current settings.')

    def _get_image(self, camera):
        self.robot.logger.info('Getting an image from: %s', camera)
        image_responses = self.image_client.get_image_from_sources([camera])

        if len(image_responses) != 1:
            print(f'Got invalid number of images: {len(image_responses)}')
            print(image_responses)
            assert False

        self.image = image_responses[0]
        if self.image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(self.image.shot.image.data, dtype=dtype)
        if self.image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(self.image.shot.image.rows, self.image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)
        return img
    
    def rotate_pixel_coordinates(self, camera, height, width, center_x, center_y):

        if camera == 'frontleft_fisheye_image' or camera == 'frontright_fisheye_image':
            rot_angle_rad = -90 * math.pi / 180
        elif camera == 'left_fisheye_image':
            rot_angle_rad = 0 * math.pi / 180
        elif camera == 'right_fisheye_image':
            rot_angle_rad = 180 * math.pi / 180
        else:
            rot_angle_rad = None
        
            
        # Since the image is rotated, x and y needs to be rotated as well
        c_x = width / 2
        c_y = height / 2

        # Translate coordinates (relative to center)
        x_rel = center_x - c_x
        y_rel = center_y - c_y

        # Perform rotation
        x_rotated = int(round(x_rel * math.cos(rot_angle_rad) - y_rel * math.sin(rot_angle_rad)))
        y_rotated = int(round(x_rel * math.sin(rot_angle_rad) + y_rel * math.cos(rot_angle_rad)))
        
        x_rotated += c_x
        y_rotated += c_y

        return x_rotated, y_rotated

    def generate_random_string(self):
        # Generate random string for pictures

        return ''.join(random.choices(string.ascii_lowercase, k=5))

    def _get_image_manipulator(self, count):
        # Take pictures with the end-effector

        image_responses = self.image_client.get_image_from_sources(['hand_color_image'])
        
        for image in image_responses:
            num_bytes = 1  # Assume a default of 1 byte encodings.
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
                extension = '.png'
            else:
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    num_bytes = 3
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    num_bytes = 4
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    num_bytes = 1
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                    num_bytes = 2
                dtype = np.uint8
                extension = '.jpg'

            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                try:
                    # Attempt to reshape array into a RGB rows X cols shape.
                    img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
                except ValueError:
                    # Unable to reshape the image data, trying a regular decode.
                    img = cv2.imdecode(img, -1)
            else:
                img = cv2.imdecode(img, -1)

            # Save the image from the GetImage request to the current directory with the filename
            # matching that of the image source.
            random_string = self.generate_random_string()
            image_saved_path = f'img/{random_string}_{count}'
            cv2.imwrite(image_saved_path + extension, img)

    def take_pictures(self, robot):
        '''
        Takes pictures when it is called. To change the picture count in every motion phase, 
        please alter PICTURE_COUNT global parameter.
        '''
        i = 0
        pic_count = PICTURE_COUNT
        while True:
            self._get_image_manipulator(i)

            if i == pic_count:
                robot.logger.info('Pictures saved.')
                break

            i += 1

            time.sleep(TRAJECTORY_DURATION / pic_count)

    def walk_to_image(self, camera):
        pallet_detector = PalletDetectionLive(self.config)
        (center_x, center_y, confidence) = pallet_detector.run(camera)

        if confidence < 0.7:
            self.robot.logger.info('No confident corner detected from stream: %s', camera)
            return False

        if center_x and center_y == 0:
            self.robot.logger.info('Could not find pallet. Aborting...')
            return False
        
        img = self._get_image(camera)

        height, width = img.shape[:2] 

        (loc_x, loc_y) = self.rotate_pixel_coordinates(camera, height, width, center_x, center_y)

        self.robot.logger.info('Location before rotation (%s, %s)', center_x,
                            center_y)
        
        self.robot.logger.info('Rotated location: (%s, %s)', int(loc_x),
                            int(loc_y))
        
        # # Draw center point
        # cv2.circle(img, (int(loc_x), int(loc_y)), 5, (255, 255, 255), -1)
        # cv2.imshow('RotatedImage', img)
        # cv2.waitKey(0) 

        walk_vec = geometry_pb2.Vec2(x=int(loc_x), y=int(loc_y))

      
        offset_distance = wrappers_pb2.FloatValue(value= DISTANCE_TO_OBJECT)

        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, transforms_snapshot_for_camera=self.image.shot.transforms_snapshot,
            frame_name_image_sensor=self.image.shot.frame_name_image_sensor,
            camera_model=self.image.source.pinhole, offset_distance=offset_distance)

        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)

        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=walk_to_request)

        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break

    def unstow_arm(self, x, y, z, roll, pitch, yaw):
        
        self.hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        (q_x, q_y, q_z, q_w) = quaternion_from_euler(roll, pitch, yaw)

        # Rotation as a quaternion
        self.flat_body_Q_hand = geometry_pb2.Quaternion(x=q_x, y=q_y, z=q_z, w=q_w)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=self.hand_ewrt_flat_body,
                                                rotation=self.flat_body_Q_hand)

        robot_state = self.robot_state_client.get_robot_state()
        self.odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = self.odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, TRAJECTORY_DURATION)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send the request
        self.cmd_id = self.command_client.robot_command(command)
        self.robot.logger.info('The arm is unstowing...')

        block_until_arm_arrives(self.command_client, self.cmd_id)

        time.sleep(2)

    def move_arm(self, x, y, z, roll, pitch, yaw):

        self.hand_ewrt_flat_body.x = x
        self.hand_ewrt_flat_body.y = y
        self.hand_ewrt_flat_body.z = z

        (q_x, q_y, q_z, q_w) = quaternion_from_euler(roll, pitch, yaw)
    
        self.flat_body_Q_hand.x = q_x
        self.flat_body_Q_hand.y = q_y
        self.flat_body_Q_hand.z = q_z
        self.flat_body_Q_hand.w = q_w

        flat_body_T_hand = geometry_pb2.SE3Pose(position=self.hand_ewrt_flat_body,
                                                 rotation=self.flat_body_Q_hand)
        odom_T_hand = self.odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, TRAJECTORY_DURATION)

        # Close the gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

        # Build the proto
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send the request
        self.cmd_id = self.command_client.robot_command(command)
        self.robot.logger.info('Moving the arm...')

        # Wait until the arm arrives at the goal.
        # Note: here we use the helper function provided by robot_command.
        self.take_pictures(self.robot)

        block_until_arm_arrives(self.command_client, self.cmd_id)

        time.sleep(1)


    def move_arm_combination(self):
        # ////////// First Motion //////////
        x = 0.8
        y = 0
        z = -0.5

        self.unstow_arm(x=x,
                        y=y,
                        z=z,
                        roll=np.deg2rad(0),
                        pitch=np.deg2rad(15),
                        yaw=np.deg2rad(0))
            
        self.move_arm(x=x,
                        y=y,
                        z=z,
                        roll=np.deg2rad(0),
                        pitch=np.deg2rad(15),
                        yaw=np.deg2rad(0))
                
        # ////////// Second Motion //////////
        x = 0.8
        y = -0.25
        z = -0.43

        self.move_arm(x=x,
                    y=y,
                    z=z,
                    roll=np.deg2rad(0),
                    pitch=np.deg2rad(15),
                    yaw=np.deg2rad(30))
        
        self.move_arm(x=x,
                    y=y,
                    z=z,
                    roll=np.deg2rad(0),
                    pitch=np.deg2rad(15),
                    yaw=np.deg2rad(45))
        
        # ////////// Third Motion //////////
        x = 0.8
        y = 0.25
        z = -0.43

        self.move_arm(x=x,
                    y=y,
                    z=z,
                    roll=np.deg2rad(0),
                    pitch=np.deg2rad(15),
                    yaw=np.deg2rad(-30))
        
        self.move_arm(x=x,
                    y=y,
                    z=z,
                    roll=np.deg2rad(0),
                    pitch=np.deg2rad(15),
                    yaw=np.deg2rad(-45))
    
        # Stow the arm
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.command_client.robot_command(stow_cmd)
        self.robot.logger.info('Stow command issued.')
        block_until_arm_arrives(self.command_client, stow_command_id, 3.0)

    def continuous_walk(self, frame_name, dx, dy, dyaw):

        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        # Build the transform for where we want the robot to be relative to where the body currently is.
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        # We do not want to command this goal in body frame because the body will move, thus shifting
        # our goal. Instead, we transform this offset to get the goal position in the output frame
        # (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # Command the robot to go to the goal point in the specified frame. The command will stop at the
        # new position.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=False))
        end_time = 10.0
        cmd_id = self.robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
        
        # Wait until the robot has reached the goal.
        while True:
            feedback = self.robot_command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                return True
            time.sleep(1)

        return True
    
    def _upload_graph_and_snapshots(self):
        """Upload the graph and snapshots to the robot."""
        print('Loading the graph from disk into local storage...')
        with open(self._upload_filepath + '/graph', 'rb') as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f'Loaded graph has {len(self._current_graph.waypoints)} waypoints and {self._current_graph.edges} edges'
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(f'{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}',
                      'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(f'{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}',
                      'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print('Uploading the graph and snapshots to the robot...')
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self.graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self.graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print(f'Uploaded {waypoint_snapshot.id}')
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self.graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print(f'Uploaded {edge_snapshot.id}')

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self.graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print('\n')
            print(
                'Upload complete! The robot is currently not localized to the map; please localize'
                ' the robot using commands (6) before attempting a navigation command.')


    def _set_initial_localization_fiducial(self):
        """Trigger localization when near a fiducial."""
        robot_state = self.robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self.graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)
        
    def _navigate_to(self, waypoint):
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            waypoint, self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
        # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        
        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False

        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self.graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f'Error while navigating {e}')
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self.graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print('Robot got lost when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print('Robot got stuck when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print('Robot is impaired.')
            return True
        else:
            # Navigation command is not complete yet.
            return False
        
    def pallet_barcode_detection(self):
        pallet_radius = 0.884 # [m]
        rotation_radius = DISTANCE_TO_OBJECT + pallet_radius
        result = False

        for i in range(5):
            corner_found = False
            # Walk to the corner
            image_control = self.walk_to_image('frontright_fisheye_image')

            if image_control is False and i == 0:
                self.robot.logger.info('Could not find pallet. Aborting...')
                break


            if image_control is not False:
                # Take pictures and save into a file
                self.move_arm_combination()

                # Run the barcode scanner
                barcode_processor = BarcodeScanner(self.image_folder, debug=False)
                barcode_processor.process_images()

                result = barcode_processor.print_most_common_barcode()
                if result:
                    removing_files = glob.glob(self.image_folder + '/*.jpg')
                    for i in removing_files:
                        os.remove(i)

                    break
                else:
                    removing_files = glob.glob(self.image_folder + '/*.jpg')
                    for i in removing_files:
                        os.remove(i)
        
            # Walk to the corner
            image_control = self.walk_to_image('right_fisheye_image')

            if image_control is not False:
                time.sleep(2)
                # Take pictures and save into a file
                self.move_arm_combination()

                # Run the barcode scanner
                barcode_processor = BarcodeScanner(self.image_folder, debug=False)
                barcode_processor.process_images()

                result = barcode_processor.print_most_common_barcode()
                if result:
                    removing_files = glob.glob(self.image_folder + '/*.jpg')
                    for i in removing_files:
                        os.remove(i)

                    break
                else:
                    removing_files = glob.glob(self.image_folder + '/*.jpg')
                    for i in removing_files:
                        os.remove(i)

                corner_found = True


                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=-0.5,
                                    dy=0.0,
                                    dyaw=0.0)

                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=0.0,
                                    dy=-(rotation_radius+INCREASE_TRAJECTORY_DISTANCE),
                                    dyaw=0.0)
                
                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=rotation_radius,#+1.0,
                                    dy=0.0,
                                    dyaw=np.deg2rad(95))

            # Walk to the corner
            image_control = self.walk_to_image('left_fisheye_image')

            if image_control is not False:
                time.sleep(2)
                # Take pictures and save into a file
                self.move_arm_combination()

                # Run the barcode scanner
                barcode_processor = BarcodeScanner(self.image_folder, debug=False)
                barcode_processor.process_images()

                result = barcode_processor.print_most_common_barcode()
                if result:
                    removing_files = glob.glob(self.image_folder + '/*.jpg')
                    for i in removing_files:
                        os.remove(i)

                    break
                else:
                    removing_files = glob.glob(self.image_folder + '/*.jpg')
                    for i in removing_files:
                        os.remove(i)

                corner_found = True

                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=-0.5,
                                    dy=0.0,
                                    dyaw=0.0)

                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=0.0,
                                    dy=(rotation_radius+INCREASE_TRAJECTORY_DISTANCE),
                                    dyaw=0.0)
                
                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=rotation_radius,#+1.0,
                                    dy=0.0,
                                    dyaw=-np.deg2rad(95))
                
            if corner_found is False:
                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=-0.5,
                                    dy=0.0,
                                    dyaw=0.0)

                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=0.0,
                                    dy=(rotation_radius+INCREASE_TRAJECTORY_DISTANCE),
                                    dyaw=0.0)
                
                self.continuous_walk(TRAJECTORY_FRAME,
                                    dx=rotation_radius,#+1.0,
                                    dy=0.0,
                                    dyaw=-np.deg2rad(95))

    def run(self, config):

        self._setup_robot()

        self.set_gripper_camera_params(config)

        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
            self.robot.logger.info('Powering on robot... This may take a several seconds.')
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), 'Robot power on failed.'
            self.robot.logger.info('Robot powered on.')

            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

            self.robot.logger.info('Commanding robot to stand...')
            blocking_stand(self.command_client, timeout_sec=10)
            self.robot.logger.info('Robot is standing.')

            time.sleep(1)

            # Upload the graph
            self._upload_graph_and_snapshots()
            # Locate Spot relative to one Fiducial tag
            self._set_initial_localization_fiducial()


            self.robot.logger.info('Navigating to the first pallet...')
            # Navigate to waypoint
            self._navigate_to(self.pallet_1)

            self.robot.logger.info('Barcode detection is starting...')
            # Start bar-code detection algorithm
            self.pallet_barcode_detection()

            self.robot.logger.info('Navigating to the second pallet...')
            # Navigate to waypoint
            self._navigate_to(self.pallet_2)

            self.robot.logger.info('Barcode detection is starting...')
            # Start bar-code detection algorithm
            self.pallet_barcode_detection()

            self.robot.logger.info('Navigating to the third pallet...')
            # Navigate to waypoint
            self._navigate_to(self.pallet_3)

            self.robot.logger.info('Barcode detection is starting...')
            # Start bar-code detection algorithm
            self.pallet_barcode_detection()

            self.robot.logger.info('Navigating to home position...')
            self._navigate_to(self.home)
                
            self.robot.logger.info('Process done.')
            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            self.robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not self.robot.is_powered_on(), 'Robot power off failed.'
            self.robot.logger.info('Robot safely powered off.')

            self.robot.logger.info('Finished.')
    
def arg_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{repr(x)} not a number')
    return x

def main(argv):
    """Command line interface."""
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-source', help='Get image from source(s)', default='frontleft_fisheye_image')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=None, type=arg_float)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help='Time [ms] to wait before the next capture',
                        type=int, default=100)
    parser.add_argument('-r', '--resize-ratio', help='Fraction to resize the image', type=float,
                        default=1)
    parser.add_argument(
        '--disable-full-screen',
        help='A single image source gets displayed full screen by default. This flag disables that.',
        action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true', default= True)
    
    # Camera parameters
    parser.add_argument(
        '--resolution',
        choices=['640x480', '1280x720', '1920x1080', '3840x2160', '4096x2160',
                 '4208x3120'], help='Resolution of the camera', default='4096x2160')
    parser.add_argument('--brightness', type=float, help='Brightness value, 0.0 - 1.0')
    parser.add_argument('--contrast', type=float, help='Contrast value, 0.0 - 1.0')
    parser.add_argument('--saturation', type=float, help='Saturation value, 0.0 - 1.0')
    parser.add_argument('--gain', type=float, help='Gain value, 0.0 - 1.0')
    parser.add_argument('--exposure', type=float, help='Exposure value, 0.0 - 1.0')
    parser.add_argument('--manual-focus', type=float, help='Manual focus value , 0.0 - 1.0')
    parser.add_argument('--auto-exposure', choices=['on', 'off'],
                        help='Enable/disable auto-exposure')
    parser.add_argument('--auto-focus', choices=['on', 'off'], help='Enable/disable auto-focus')
    parser.add_argument(
        '--hdr-mode', choices=['off', 'auto', 'manual1', 'manual2', 'manual3', 'manual4'], help=
        'On-camera high dynamic range (HDR) setting.  manual1-4 modes enable HDR with 1 the minimum HDR setting and 4 the maximum'
    )
    parser.add_argument('--led-mode', choices=['off', 'torch'],
                        help='LED mode. "torch": On all the time.', default='torch')
    parser.add_argument('--led-torch-brightness', type=float,
                        help='LED torch brightness value when on all the time, 0.0 - 1.0', default=1.0)
    parser.add_argument('--gamma', type=float, help='Gamma value, 0.0 - 1.0')
    parser.add_argument('--sharpness', type=float, help='Sharpness value, 0.0 - 1.0')
    parser.add_argument('--white-balance-temperature-auto', choices=['on', 'off'],
                        help='Enable/disable white-balance-temperature-auto')
    parser.add_argument('--white-balance-temperature', type=float,
                        help='Manual white-balance-temperature value , 0.0 - 1.0')

    options = parser.parse_args(argv)

    try:
        walk_to_object_instance = WalkToPallet(options)
        walk_to_object_instance.run(options)
        return True
    except Exception:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
