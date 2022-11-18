import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')

class LicensePlateDetector:
    def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None
        
        
    def detect(self, img_path: str):
        orig = cv2.imread(img_path)
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        # print(blob)
        self.net.setInput(blob)
        output_layers = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layers)
        # print(layer_outputs)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                # print(scores)
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                # print(confidence)
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                roi = img[y:y + h, x+3:x + w+3]
                text = ocr.ocr(roi, cls=False, det=False)
                plate_number=text[0][0][0]
                cv2.putText(img, plate_number, (x, y-10), self.font, 2, (0, 255, 0), 2)
                
                cv2.rectangle(img, (x,y), (x + w, y + h), self.color, 2)
                cv2.putText(img, label + ' ' + confidence, (x-100, y+50), self.font, 2, self.color, 3)
        self.fig_image = img
        self.coordinates = (x, y, w, h)

    # def ocr(self):    
    #     result = ocr.ocr(self.roi_image, cls=False, det=False)
    #     plate_number=result[0][0][0]
    #     cv2.putText(self.fig_image, plate_number, (self.coordinates[0], self.coordinates[1]-10), self.font, 5, (0, 255, 0), 3)

    #     return
    
    
    # def crop_plate(self):
    #     x, y, w, h = self.coordinates
    #     roi = self.img[y:y + h, x+3:x + w+3]
    #     self.roi_image = roi
    #     return

    def detect_video(self,video_path):
      vidcap = cv2.VideoCapture(video_path)
      fps = vidcap.get(cv2.CAP_PROP_FPS)
      print(fps)    #cv2 for getting fps
      success,frame = vidcap.read()
      count = 0
      frame_width = int(vidcap.get(3))
      frame_height = int(vidcap.get(4))
        
      size = (frame_width, frame_height)
      result = cv2.VideoWriter('results/detected.avi', 
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              10, size)

      while success:

        success,img = vidcap.read()

        if count%(fps/10)==0:   # gets one frame per second
            
            height, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)            
            # print(blob)
            self.net.setInput(blob)
            output_layers = self.net.getUnconnectedOutLayersNames()
            # print(output_layers)
            layer_outputs = self.net.forward(output_layers)
            # print(layer_outputs)
            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    # print(scores)
                    class_id = np.argmax(scores) 
                    confidence = scores[class_id]
                    # print(confidence)
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        
                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(['LP'][class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    roi = img[y:y + h, x+3:x + w+3]
                    text = ocr.ocr(roi, cls=False, det=False)
                    plate_number=text[0][0][0]
                    cv2.putText(img, plate_number, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
                    cv2.putText(img, label + ' ' + confidence, (x-100, y+50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)


            cv2.putText(img, f"Frame Number: {count+1}", (0,100), cv2.FONT_HERSHEY_COMPLEX, 1, [0,0,255], 2)
            result.write(img)     # save frame as JPEG file      
            
                
            
        count+=1
              
