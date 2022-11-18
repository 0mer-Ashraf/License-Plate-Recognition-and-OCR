# License-Plate-Recognition-and-OCR

The detector.py has a class for license plate detection and ocr. The class does all the work for you.

# Setup and Prerequisites
Download the weights in utilities folder. A check is already in place in the code which downloads the weights if they are not present. You can edit the path of the weight in the code for custom weights.
Add all testing files (images and videos) you want in the testing folder. The detections are stored in detections folder.

# RUN DETECTION ON VIDEOS
Use the following command line scripts for running the code

```
python main -v "path/to/video"  #for detection on video (might take some time).
```
# RUN DETECTION ON IMAGES

```
python main -i "path/to/image"  for detection on image (almost instantaneous).
```
