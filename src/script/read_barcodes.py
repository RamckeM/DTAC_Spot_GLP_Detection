import cv2
import math
from scipy import ndimage
from collections import Counter
import matplotlib.pyplot as plt
import os

detected_barcodes = []  # List to store detected barcodes
##  Debug flag - Set to 'True' to show images at different processing stages and print intermediate results
DEBUG = 0
##  Dictionary representing the ITF encoding
ITF_DICT = {
    '00110': 0,
    '10001': 1,
    '01001': 2,
    '11000': 3,
    '00101': 4,
    '10100': 5,
    '01100': 6,
    '00011': 7,
    '10010': 8,
    '01010': 9
}
ITF_START_CODE = '0000'
ITF_STOP_CODE = '100'
ITF_SYMBOL_LENGTH = 5

class BarcodeScanner:
    def __init__(self, folder_path, debug=False) -> None:
        self.folder_path = folder_path
        self.detected_barcodes = []
        self.debug = debug

    def detect(self, img):  
        ##  Create barcode object and use detection function
        bardet = cv2.barcode.BarcodeDetector()
        detected, points = bardet.detect(img)

        if not detected:
            # print("NO BARCODE DETECTED")
            return None

        if self.debug:
            print("{}".format(points[0]))
            img_lines = img.copy()
            img_lines = cv2.polylines(img_lines, points.astype(int), True, (0, 255, 0), 3)
            plt.figure()
            plt.imshow(img_lines)

        return points[0]

    def calc_angle(self, points):
        ## Define upper corners of the rectangle
        px1, py1 = int(points[1][0]), int(points[1][1])
        px2, py2 = int(points[2][0]), int(points[2][1])
        angle = math.degrees(math.atan2(py2-py1, px2-px1))

        if self.debug:
            print("{}".format(angle))

        return angle


    def crop_img(self, img, points, eps):
        ##  Determine corner points
        y1 = int(max(points[1][1], points[2][1]) + (eps*img.shape[1]))
        y2 = int(min(points[0][1], points[3][1]) - (eps*img.shape[1]))
        x1 = int(max(points[0][0], points[1][0]) + (eps*img.shape[0]))
        x2 = int(min(points[2][0], points[3][0]) - (eps*img.shape[0]))

        img_cropped = img[y1:y2, x1:x2]

        if self.debug:
            plt.figure()
            plt.imshow(img_cropped)
        
        return img_cropped


    ##  Expecting array containing a bit array
    def decode(self, line):
        ##  Assign the bit values to individual bars by determining their width and threshold change
        bars = []
        current_length = 1
        for i in range(len(line)-1):
            if line[i] == line[i+1]:
                current_length += 1
            else:
                bars.append(current_length)
                current_length = 1
        ##  Append the last bar manually
        bars.append(current_length)

        ##  Remove quite zones
        try:
            bars.pop(0)
            bars.pop(len(bars)-1)
        except:  # noqa: E722
            print("DecoderError")
            return None
        
        avg_bar = (sum(bars)/len(bars))

        if self.debug:
            print("no. of bars: {}".format(len(bars)))
            print("avg. length of bar: {}".format(avg_bar))
            [print(bar, end=' ') for bar in bars]
            print("\n")
        
        ##  Remap the bar width to bits (narrow=0, wide=1)
        for i in range(len(bars)):
            if bars[i] > avg_bar:
                bars[i] = 1
            else:
                bars [i] = 0

        ##  Flip the bit list if the start code was not detected at the beginning
        if ''.join(map(str, bars[:len(ITF_START_CODE)])) != ITF_START_CODE:
            bars.reverse()

        ##  Remove the start and stop code and split the list according to colour
        bars = bars[len(ITF_START_CODE):len(bars)-len(ITF_STOP_CODE)+1]
        black_bars = bars[::2]
        white_bars = bars[1::2]

        ##  Iterate through all bars and decode in groups of five
        code = []
        for i in range(int(len(black_bars)/ITF_SYMBOL_LENGTH)):
            ##  Parse through the bars with step sizes equal to the itf symbol size (5)
            index = i*ITF_SYMBOL_LENGTH
            black_bit_pattern = ''.join(map(str, black_bars[index:index+ITF_SYMBOL_LENGTH]))
            white_bit_pattern = ''.join(map(str, white_bars[index:index+ITF_SYMBOL_LENGTH]))
            ##  Map the pattern to the dict and append the respective digit to a new list
            try:
                code.append(ITF_DICT[black_bit_pattern])
                code.append(ITF_DICT[white_bit_pattern])
            except:  # noqa: E722
                # print("KeyError")
                return None

        ##  Remap code list to string and return
        code_str = ''.join(map(str, code))
        return code_str
    
    def process_images(self):
        # Iterate over all files in the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                try:
                    # Load the image
                    img_path = os.path.join(self.folder_path, filename)
                    img = cv2.imread(img_path)

                    # Perform barcode detection
                    points = self.detect(img)
                    angle = self.calc_angle(points)
                    img_rotated = ndimage.rotate(img, angle)

                    # Pre-crop the image based on initial detection
                    points2 = self.detect(img_rotated)
                    img_interm = self.crop_img(img_rotated, points2, -0.1)

                    # Determine barcode position in the pre-processed image
                    points3 = self.detect(img_interm)
                    img_cropped = self.crop_img(img_interm, points3, .005)

                    # Convert to grayscale, blur, and apply threshold filter for binarization
                    gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Bit inversion for more intuitive array handling
                    inv = cv2.bitwise_not(thresh)

                    # Iterate over all vertical lines of the image and decode them individually
                    results = []
                    for i in range(inv.shape[0]):
                        scan_line = inv[i]
                        scan_line[scan_line == 255] = 1
                        j = 0
                        while scan_line[j] == 1:
                            scan_line[j] = 0
                            j += 1
                        j = len(scan_line) - 1
                        while scan_line[j] == 1:
                            scan_line[j] = 0
                            j -= 1
                        results.append(self.decode(scan_line))

                    # Print results for each image
                    print("File: {}, Barcode: {}".format(filename, Counter(results).most_common(1)[0][0]))
                    # Add detected barcodes to the list
                    self.detected_barcodes.extend(results)
                except Exception:
                    # print("Error processing {}: {}".format(filename, e))
                    continue

    def print_most_common_barcode(self):
        # Count occurrences of each barcode
        barcode_counts = Counter(self.detected_barcodes)
        # Print the most repeated barcode
        most_common_barcode = barcode_counts.most_common(1)
        if most_common_barcode:
            if most_common_barcode[0][0] == 0 or most_common_barcode[0][0] is None:
                print("No barcodes detected.")
                return False
            else:
                print("Most repeated barcode:", most_common_barcode[0][0])
                return True
        else:
            print("No barcodes detected.")
            return False