import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolo\yolov3.weights", "yolo\yolov3.cfg")
classes = []
with open("yolo\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("img\mobil.jpg")
# img = cv2.resize(img, None, fx=0.3, fy=0.3)
vehicle = {'car', 'bicycle', 'motorbike', 'bus', 'truck'}
vehicle_count = {
    'car': 0,
    'bicycle': 0,
    'motorbike': 0,
    'bus': 0,
    'truck': 0,
}
imaginer_line = 350

scale_percent = (720 * 100) / img.shape[0]  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
height, width, channels = img.shape

color = (0, 255, 0)
thickness = 2
cv2.line(img, (0, imaginer_line), (width, imaginer_line), color, thickness)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        if label in vehicle:
            # print(label)
            if label == 'car':
                vehicle_count['car'] += 1
            elif label == 'bicycle':
                vehicle_count['bicycle'] += 1
            elif label == 'motorbike':
                vehicle_count['motorbike'] += 1
            elif label == 'bus':
                vehicle_count['bus'] += 1
            else:
                vehicle_count['truck'] += 1
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

print(vehicle_count)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
