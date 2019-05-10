import numpy as np
from scipy import ndimage
import cv2
import pytesseract
from PIL import Image
from pytesseract import Output
from googletrans import Translator
from pytesseract import image_to_boxes
from pytesseract import image_to_data
import os

cap = cv2.VideoCapture(0)
frame_counter = 0
translator = Translator()

# This function figures out how large a given string can be rendered to fit in a given bounding box
def drawString(frame, s, x, y, width, height):
    scale = 0
    size = 0
    while True:
        b = cv2.getTextSize(s, cv2.FONT_HERSHEY_COMPLEX, scale + 0.01, 1)
        b = b[0]
        if(b[0] > width or b[1] > height):
            break
        scale = scale + 0.01
        size = b
    cv2.putText(frame, s, (x, y + height), cv2.FONT_HERSHEY_COMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)

# load data from the images directory
directory = "./images"

# this will go though all images in the images directory and run OCR and optionally translate on each image,
# before saving the converted image out to the results directory
for filename in os.listdir(directory):
    print("opening " + filename)
    frame = np.array(Image.open(os.path.join(directory, filename)))
    # Optionally read input from webcam instead of loading a file
    # ret, frame = cap.read()
    key = cv2.waitKey(1) & 255

    # do various preprocessing filter steps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 3)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.bilateralFilter(img, 9, 75, 75)
    im = np.array(img)

    # extract the detected text and bounding boxes
    data = image_to_data(im, output_type=Output.DICT)

    n = len(data['level'])

    count = 0

    # we first go through all of the found words to paint a rectangle over them, so we can draw new text on top of them
    for i in range(n):
        (x_1, y_1, w, h, text) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['text'][i].encode("utf-8"))
        if (len(text.decode("utf-8").strip()) > 0):
            y_2 = y_1 + h
            x_2 = x_1 + w
            count = count + 1
            # Instead of drawing a constant color rectangle, we approximate the color of the background here by taking the median of the rectangle
            average0 = np.median(frame[y_1:y_2, x_1:x_2, 0])
            average1 = np.median(frame[y_1:y_2, x_1:x_2, 1])
            average2 = np.median(frame[y_1:y_2, x_1:x_2, 2])
            cv2.rectangle(frame, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (average0, average1, average2), -1)
    x = 0
    y = 0
    sx = 0
    height = 0
    s = ""
    print("NUMBER OF RESULTS: " + str(count))
    # now go though again, this time with some logic to join words into longer sentences or sentence fragments in order to be able to translate them more coherently
    for i in range(n):
        (x_1, y_1, w, h, text) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['text'][i].encode("utf-8"))
        if len(text.strip()) == 0:
            continue
        # if this word is close enough to the previous word, include it in the same sentence
        if x_1 - x < w / 2 and abs(y_1 - y) < h and abs(height - h) < h:
            s = s + " " + text
            x = x_1 + w
            height = max(h, height)
        # otherwise handle the previous sentence and start a new one
        else:
            s = s.strip()
            if len(s) > 0:
                # This line will translate the detected string, while the next will leave it in english
                # translated = translator.translate(s, dest="es", src="en").text.encode("ascii", errors="ignore")
                translated = s.decode("ascii", errors="ignore")
                
                width = x - sx
                drawString(frame, translated, sx, y, width, height)
            s = text
            sx = x_1
            x = sx + w
            height = h
            y = y_1
    s = s.strip()

    # there might be a final string that doesn't get caught in the for loop so we
    # do one last iteration manually
    if len(s) > 0:
        # This line will translate the detected string, while the next will leave it in english
        # translated = translator.translate(s, dest="es", src="en").text.encode("ascii", errors="ignore")
        translated = s.decode("ascii", errors="ignore")
        
        width = x - sx
        drawString(frame, translated, sx, y, width, height)
    print("translated")
    # cv2.imshow('frame', frame)

    # Save the image to the results directory
    path = os.path.splitext(filename)        
    print("saving " + os.path.join("results", path[0] + "_translated" + path[1]))
    Image.fromarray(frame).save(os.path.join("results", path[0] + "_translated" + path[1]))
    # if key == ord('q'):
        # break

# cv2.waitKey(-1)
cap.release()
# cv2.destroyAllWindows()