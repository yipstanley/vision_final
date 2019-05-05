import numpy as np
from scipy import ndimage
import cv2
import pytesseract
from PIL import Image
from pytesseract import Output
# from matplotlib.pyplot import imread
# from matplotlib.pyplot import imread
from pytesseract import image_to_boxes
from autocorrect import spell

cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
i = 5
frame_counter = 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # cv2.rectangle(img, (267, 256), (267+50, 256+26), (0, 255, 0), 10)
    im = np.array(img)
    # saveim = Image.fromarray(im)
    # saveim.save(str(frame_counter) + ".jpg", "JPEG")
    # openedim = Image.open(str(frame_counter) + ".jpg")
    # if (i == 5):
        # data = i mage_to_boxes(img, lang="eng", output_type=Output.DICT)
    data = image_to_boxes(im, lang="eng", output_type=Output.DICT)

        # if len(string) > 0:
    #     # #     # print(spell(string))
    #     #     print(string)
    #     i = 0
    # else:
    #     i += 1
    
    print(data)
    for i in range(len(data['left'])):
        (x_1, y_1, x_2, y_2) = (data['left'][i], data['top'][i], data['right'][i], data['bottom'][i])
        y_1 = img.shape[0] - y_1
        y_2 = img.shape[0] - y_2
        print("XYWH:")
        print(x_1)
        print(y_1)
        print(x_2)
        print(y_2)
        cv2.rectangle(im, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 255, 0), 5)
    cv2.imshow('frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()