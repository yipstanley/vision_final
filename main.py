import numpy as np
from scipy import ndimage
import cv2
import pytesseract
from PIL import Image
from pytesseract import image_to_string

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    print(image_to_string(Image.fromarray(img), lang="eng").encode('utf-8'))
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()