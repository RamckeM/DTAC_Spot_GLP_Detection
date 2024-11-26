import argparse
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient



class RobotPose():

    def __init__(self) -> None:

        # bosdyn client arguments
        parser = argparse.ArgumentParser()
        bosdyn.client.util.add_base_arguments(parser)
        options = parser.parse_args()

        # Create robot object
        sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
        robot = sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(robot)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    def publish_information(self):
        '''Robot odom and vision information publisher'''

        delay = 0.5
        running = True
        while running:
            robot_position = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot.child_to_parent_edge_map

            print(f"odom information:\n {robot_position['odom']}")
            print(f"vision information:\n {robot_position['vision']}")

            time.sleep(delay)
            
        
if __name__ == '__main__':
    robot = RobotPose()
    robot.publish_information()