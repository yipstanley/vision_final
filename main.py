import numpy as np
from scipy import ndimage
import cv2
import pytesseract
from PIL import Image
from pytesseract import Output
from googletrans import Translator
# from matplotlib.pyplot import imread
# from matplotlib.pyplot import imread
from pytesseract import image_to_boxes
from pytesseract import image_to_data
from autocorrect import spell

cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
i = 5
frame_counter = 0
translator = Translator()

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 3)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # cv2.rectangle(img, (267, 256), (267+50, 256+26), (0, 255, 0), 10)
    im = np.array(img)
    # saveim = Image.fromarray(im)
    # saveim.save(str(frame_counter) + ".jpg", "JPEG")
    # im = Image.open("test3.jpg")
    # im = np.array(im)
    # if (i == 5):
        # data = i mage_to_boxes(img, lang="eng", output_type=Output.DICT)
    data = image_to_data(im, output_type=Output.DICT)
    n = len(data['level'])

        # if len(string) > 0:
    #     # #     # print(spell(string))
    #     #     print(string)
    #     i = 0
    # else:
    #     i += 1

    # print(data)
    for i in range(n):
        (x_1, y_1, w, h, text) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['text'][i].encode("utf-8"))
        # y_1 = img.shape[0] - y_1
        # y_2 = img.shape[0] - y_2
        if (len(text.decode("utf-8").strip()) > 0):
            y_2 = y_1 + h
            x_2 = x_1 + w
            print(text)
            # print("XYWH:")
            # print(x_1)
            # print(y_1)
            # print(x_2)
            # print(y_2)
            average0 = np.median(frame[y_1:y_2, x_1:x_2, 0])
            average1 = np.median(frame[y_1:y_2, x_1:x_2, 1])
            average2 = np.median(frame[y_1:y_2, x_1:x_2, 2])
            cv2.rectangle(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (average0, average1, average2), -1)
    for i in range(n):
        (x_1, y_1, w, h, text) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['text'][i].encode("utf-8"))
        if (len(text.decode("utf-8").strip()) > 0):
            y_2 = y_1 + h
            x_2 = x_1 + w
            # translated = "VELOCIDAD" if (text == "SPEED") else "L?MITE" 
            # translated = translator.translate(text, dest="es", src="en").text.encode("utf-8")
            # print(type(text))
            cv2.putText(frame, text.decode("utf-8"), (x_1, y_2), cv2.FONT_HERSHEY_COMPLEX, max((x_2 - x_1) / 200, 0.3), (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(im, text.decode("utf-8"), (x_1, y_2), cv2.FONT_HERSHEY_COMPLEX, max((x_2 - x_1) / 200, 0.3), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()