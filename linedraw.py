import cv2
import numpy as np

img = cv2.imread("img\mobil.jpg")
print('Original Dimensions : ', img.shape)
scale_percent = (720 * 100) / img.shape[0]  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
first_segment = 250
last_segment = 600
line_segment = int((600-250)/3)

resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', resized.shape)
# start_point =
end_point = (250, 250)
color = (0, 255, 0)
thickness = 2
cv2.line(resized, (0, first_segment), (width, first_segment), color, thickness)
cv2.line(resized, (0, first_segment+line_segment), (width, first_segment+line_segment), color, thickness)
cv2.line(resized, (0, first_segment+line_segment*2), (width, first_segment+line_segment*2), color, thickness)
cv2.line(resized, (0, last_segment), (width, last_segment), color, thickness)
# cv2.line(resized, (0, line_segment*3), (width, line_segment*3), color, thickness)
cv2.imshow("Test", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
