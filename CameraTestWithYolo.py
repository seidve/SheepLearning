import cv2
import numpy as np

# Load yolov3 tiny pretrained with coco
yolo = cv2.dnn.readNet("./yolo/yolov3-tiny.weights", "./yolo/darknet-master/cfg/yolov3-tiny.cfg")

# Read class names from coco data
classes = []
with open("./yolo/darknet-master/data/coco.names", 'r') as f:
    classes = f.read().splitlines()

# Read test image and get size
vid = cv2.VideoCapture(0)
W, H = 1920, 1080
vid.set(cv2.CAP_PROP_FRAME_WIDTH, W)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

fps = vid.get(cv2.CAP_PROP_FPS)

print("fps: " + str(fps))

colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

while True:

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture the video frame by frame
    ret, img = vid.read()

    height, width, _ = img.shape

    resized = cv2.resize(img, (320, 320))

    # Preprocess image:
    blob = cv2.dnn.blobFromImage(resized, 1 / 255, (320, 320), (0, 0, 0), swapRB=False, crop=False)

    # Use yolo to analyze image
    yolo.setInput(blob)

    output_layer_names = yolo.getUnconnectedOutLayersNames()
    layer_output = yolo.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_output:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            # Get bounding box and class if confidence over .7
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
    # counter = 0
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            # if label.lower() != "sheep":
            #     continue

            # counter += 1
            config = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + config, (x, y + 20), font, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Yolo Camera Test', img)

vid.release()
cv2.destroyAllWindows()
