from Detector import LicensePlateDetector
import argparse
from os.path import exists
import gdown
import matplotlib.pyplot as plt
import cv2

parser= argparse.ArgumentParser(description="DETECT PLATE")
parser.add_argument("-i",'--image', type=str, help="Path for the image")
parser.add_argument("-v","--video",type=str,help="Path for video")
args=parser.parse_args()

if exists("utilities/lapi.weights")==False:
    url="https://drive.google.com/u/0/uc?id=105RNX2V4sKIoPgu3P6hTkZxhanjFeXLs&export=download"
    output="utilities/lapi.weights"
    gdown.download(url, output, quiet=False)
    
weights_path = 'utilities/lapi.weights'
configuration_path = 'utilities/darknet-yolov3.cfg'
classes_path="utilities/classes.names"

lpd = LicensePlateDetector(
    pth_weights=weights_path, 
    pth_cfg=configuration_path, 
    pth_classes=classes_path
)

# Detect license plate.
if args.image:
    lpd.detect(args.image)

# Plot image with detected plate and OCR
    plt.figure(figsize=(24, 24))
# plt.imshow(cv2.cvtColor(lpd.fig_image, cv2.COLOR_BGR2RGB))
    plt.savefig('results/detected.jpg')
# plt.show()

# detection on video
if args.video:
    lpd.detect_video(args.video)