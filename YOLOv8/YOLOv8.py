import numpy
import os
import cv2
import yaml
from ultralytics import YOLO

class YOLOv8:
    def __init__(self):
        self.model = None
        self.class_mapping = None

    def loadModel(self,modelFolder,modelName=None):
        self.model = YOLO(os.path.join(modelFolder,"train/weights/best.pt"))
        with open(os.path.join(modelFolder,"class_mapping.yaml"),"r") as f:
            self.class_mapping = yaml.safe_load(f)

    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,0)
        results = self.model.predict(image)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        class_nums = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        resultList = []
        print(bboxes.shape)
        for i in range(bboxes.shape[0]):
            class_value = class_nums[i]
            class_name = self.class_mapping[class_value]
            xmin,ymin,xmax,ymax = bboxes[i]
            confidence = confs[i][class_value]
            bbox = {"class":class_name,
                    "xmin":xmin,
                    "ymin":ymin,
                    "xmax":xmax,
                    "ymax":ymax,
                    "conf":confidence}
            resultList.append(bbox)
        return str(resultList)

    def createModel(self):
        self.model = YOLO("yolov8s.pt")
        return self.model