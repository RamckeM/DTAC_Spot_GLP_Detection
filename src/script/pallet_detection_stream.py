import argparse
import sys
import time

import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from ultralytics import YOLO

VALUE_FOR_Q_KEYSTROKE = 113
VALUE_FOR_ESC_KEYSTROKE = 27

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -90,  # -78 for exact rotation
    'frontright_fisheye_image': -90, # -102 for exact rotation
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

class PalletDetectionLive:
    def __init__(self, options) -> None:
        self.options = options
        self.model = YOLO("/res/best.pt")
        self.image_client = None

    def connect_spot(self):
        # Create robot object with an image client.
        sdk = bosdyn.client.create_standard_sdk('image_capture')
        robot = sdk.create_robot(self.options.hostname)
        bosdyn.client.util.authenticate(robot)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()

        self.image_client = robot.ensure_client(self.options.image_service)

    def process_image(self, image):
        dtype = np.uint8
        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.options.auto_rotate:
            img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name], reshape=False)

        center_x = 0
        center_y = 0
        max_confidence = 0
        max_confidence_box = None

        try:
            # Perform YOLO detection
            detections = self.model(img)[0]
            
            print("Detections:", detections.boxes.data.tolist())

            # Process detections and draw bounding boxes
            for data in detections.boxes.data.tolist():
                xmin, ymin, xmax, ymax, confidence, class_id = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
                if class_id == 0:
                    continue
                # Draw bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # Draw confidence score
                cv2.putText(img, str(round(confidence, 2)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                # Check if current detection has higher confidence than previous maximum
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_confidence_box = data

            # Draw center point for the box with maximum confidence, if it exists
            if max_confidence_box is not None:
                # Retrieve coordinates of the rectangle with the highest confidence
                xmin, ymin, xmax, ymax, _, _ = map(int, max_confidence_box[:6])
                # Calculate center coordinates
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)
                # Draw center point
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
                # Display confidence score of the max confidence box
                cv2.putText(img, f"Max Confidence: {round(max_confidence, 2)}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

        except Exception as e:
            print("Error occurred during YOLO detection:", e)

        return img, center_x, center_y, max_confidence
    
    def run(self, camera):
        self.connect_spot()

        requests = [
        build_image_request(camera, quality_percent=self.options.jpeg_quality_percent,
                            resize_ratio=self.options.resize_ratio)]
        
        print("\nCorner detection is starting, this might take a while...\n")
        print("Camera: {}".format(camera))
        
        t1 = time.time()
        image_count = 0
        while image_count != 3:
        
            images_future = self.image_client.get_image_async(requests, timeout=0.5)
            
            # time.sleep(1.0)

            images = images_future.result()

            for i in range(len(images)):
                processed_image, center_x, center_y, confidence = self.process_image(images[i])
                cv2.imshow(images[i].source.name, processed_image)
                print("Detected image center x= {}, y={}".format(center_x, center_y))
                
            cv2.waitKey(self.options.capture_delay)
            image_count += 1
            print(f'Mean image retrieval rate: {image_count/(time.time() - t1)}Hz')

        return center_x, center_y, confidence

def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-source', help='Get image from source(s)', default='frontleft_fisheye_image')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
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
    options = parser.parse_args(argv)

    pallet_detector = PalletDetectionLive(options)
    pallet_detector.run()

if __name__ == '__main__':
    main(sys.argv[1:])