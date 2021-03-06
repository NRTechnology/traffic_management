# import cv2
# import numpy as np

# Load Yolo
# net = cv2.dnn.readNet("yolo\yolov3.weights", "yolo\yolov3.cfg")
# classes = []
# with open("yolo\coco.names", "r") as f:
#    classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()

# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
# img = cv2.imread("img\mobil.jpg")
# img = cv2.resize(img, None, fx=0.3, fy=0.3)
# vehicle = {'car', 'bicycle', 'motorbike', 'bus', 'truck'}
# vehicle_count = {
#    'segment1': [0, 0, 0, 0, 0],
#    'segment2': [0, 0, 0, 0, 0],
#    'segment3': [0, 0, 0, 0, 0],
# }
# first_segment = 250
# last_segment = 600
# line_segment = int((600 - 250) / 3)

# scale_percent = (720 * 100) / img.shape[0]  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)

# img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# height, width, channels = img.shape

# color = (0, 255, 0)
# thickness = 2
# cv2.line(img, (0, first_segment), (width, first_segment), color, thickness)
# cv2.line(img, (0, first_segment + line_segment), (width, first_segment + line_segment), color, thickness)
# cv2.line(img, (0, first_segment + line_segment * 2), (width, first_segment + line_segment * 2), color, thickness)
# cv2.line(img, (0, last_segment), (width, last_segment), color, thickness)

# Detecting objects
# blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# net.setInput(blob)
# outs = net.forward(output_layers)

# Showing informations on the screen
# class_ids = []
# confidences = []
# boxes = []
# for out in outs:
#    for detection in out:
#        scores = detection[5:]
#        class_id = np.argmax(scores)
#        confidence = scores[class_id]
#        if confidence > 0.5:
#            # Object detected
#            center_x = int(detection[0] * width)
#            center_y = int(detection[1] * height)
#            w = int(detection[2] * width)
#            h = int(detection[3] * height)
#            # Rectangle coordinates
#            x = int(center_x - w / 2)
#            y = int(center_y - h / 2)

#            boxes.append([x, y, w, h])
#            confidences.append(float(confidence))
#            class_ids.append(class_id)

# indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# font = cv2.FONT_HERSHEY_PLAIN
# for i in range(len(boxes)):
#    if i in indexes:
#        x, y, w, h = boxes[i]
#        label = str(classes[class_ids[i]])
#        if label in vehicle:
#            # print(label)
#            vehicle_count['segment1'][0] += 1
#        color = colors[i]
#        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#        # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

# print(vehicle_count)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
