## Demo source: https://www.youtube.com/watch?v=b59xfUZZqJE

import cv2
import numpy as np
import matplotlib.pyplot as plt

## load yolov3 tiny pretrained with coco
yolo = cv2.dnn.readNet("./yolo/yolov3-tiny.weights", "./yolo/darknet-master/cfg/yolov3-tiny.cfg")

## read class names from coco data
classes = []
with open("./yolo/darknet-master/data/coco.names", 'r') as f:
    classes = f.read().splitlines()

## read test image and get size
img = cv2.imread("./images/schafe.jpg")
resized = cv2.resize(img, (320, 320))
height, width, _ = img.shape
# cv2.imshow('test',img)

## preprocess image:
blob = cv2.dnn.blobFromImage(resized, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

## use yolo to analyze image
yolo.setInput(blob)

output_layer_names = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layer_names)

boxes = []
confidences = []
class_ids = []

for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        # get bounding box and class if confidence over .7
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]

        label = str(classes[class_ids[i]])
        if label != "sheep":
            continue

        config = str(round(confidences[i], 2))
        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + config, (x, y + 20), font, 1, (255, 255, 255), 2)

plt.imshow(img)
plt.show()
