# import libraries here
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
from matplotlib import pyplot as plt
import os
import cv2
import math
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import sys

import pyocr
import pyocr.builders

import matplotlib
import matplotlib.pyplot as plt


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """
    def __init__(self, birth_date: str, expiry_date: str, gender: str, name: str, surname: str,number: str):
        self.birth_date = birth_date
        self.expiry_date = expiry_date
        self.gender = gender
        self.name = name
        self.surname = surname
        self.number = number

    def fit(self, text):
        text_list = text.splitlines()
        print(text_list)
        if len(text_list) < 2:
            return self
        first = True
        for t in text_list:
            if len(t) < 10:
                continue
            if first:
                x = t.find('SRB')
                y = t.find('<',x)
                if x == -1 or y == -1:
                    break
                self.name = t[x+3:y]
                z = t.find('<',y+2)
                if z == -1:
                    break
                self.surname = t[y+1: z]
                first = False
            else:
                x = t.find('SRB')
                if x == -1:
                    break
                crr = x - 10
                if crr < 0:
                    crr = 0
                self.number = t[crr:x]
                y = t.find('F')
                if y == -1:
                    self.gender = 'M'
                    y = t.find('M')
                    if y == -1:
                        break
                else:
                    self.gender = 'F'
                temp = t[x+3:x+3+6]
                self.birth_date = self.format_date(temp, "19")

                temp = t[y+1: y+1+6]
                self.expiry_date = self.format_date(temp, "20")
                break;
        return self
    def format_date(self, date, type):
        year = str(""+type + date[:2])
        month = date[2:4]
        day = date[4:6]
        return day + "." + month + "." + year

def extract_information_from_image(image_path) -> Person:
    """
    Procedura prima putanju do slike sa koje treba izvuci informacije.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Person>  Informacije o prepoznatoj osobi
    """
    person = Person(None, None, None, None, None, None)
    # TODO - Prepoznati sve podatke o osobi sa fotografije pasosa (datum rodjenja, datum isteka pasosa, pol (M/F), ime, prezime, broj pasosa)
    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    # predictor = dlib.shape_predictor('haarcascade_frontalface_default.xml')
    print("Start")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    #test(image_path)

    # ucitavanje i transformacija slike
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    screen = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #img = face_alignment(image_path)
    #plt.imshow(img, 'gray')
    #plt.show()

    # detekcija svih lica na grayscale slici
    faces = face_cascade.detectMultiScale(screen, 1.35, 5)
    print("End")

    new_img = image
    # iteriramo kroz sve detekcije korak 1.
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        # filtering small imange if it gets caught first
        if w < 150:
            continue
        roi_color = screen[y:y + h, x:x + w]

        img = screen[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(img)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

        #   plt.imshow(roi_color, 'gray')
        #   plt.imshow(processed_img, 'gray')
        #   plt.imshow(img, 'gray')
        #   plt.show()

        if len(eyes) >= 2:
            eye = eyes[:, 2]
            container1 = []
            for i in range(0, len(eyes)):
                container = (eyes[i][0] * eyes[i][1], i)
                container1.append(container)
            df = pd.DataFrame(container1, columns=[
                "length", "idx"]).sort_values(by=['length'])
            eyes = eyes[df.idx.values[0:2]]

            # deciding to choose left and right eye
            eye_1 = eyes[0]
            eye_2 = eyes[1]
            if eye_1[0] > eye_2[0]:
                left_eye = eye_2
                right_eye = eye_1
            else:
                left_eye = eye_1
                right_eye = eye_2

            # center of eyes
            # center of right eye
            right_eye_center = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)))
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]
            cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

            # center of left eye
            left_eye_center = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]
            cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

            # finding rotation direction
            if left_eye_y > right_eye_y:
                print("Rotate image to clock direction")
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate image direction to clock
            else:
                print("Rotate to inverse clock direction")
                point_3rd = (right_eye_x, left_eye_y)
                direction = 1  # rotate inverse direction of clock

            cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
            a = trignometry_for_distance(left_eye_center,
                                         point_3rd)
            b = trignometry_for_distance(right_eye_center,
                                         point_3rd)
            c = trignometry_for_distance(right_eye_center,
                                         left_eye_center)
            # fix dividing by zero
            if 2*b*c == 0:
                c = 1
                b = 1
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = (np.arccos(cos_a) * 180) / math.pi

            if direction == -1:
                angle = 90 - angle
            else:
                angle = -(90 - angle)

            angle = -angle
            # rotate image
            new_img = image
            h, w = new_img.shape[:2]
            print(h, w, angle)
            plt.imshow(new_img, 'gray')
            #   plt.imshow(processed_img, 'gray')
            #   plt.imshow(img, 'gray')
            #   plt.show()
            m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            new_img = cv2.warpAffine(image, m, (w, h))
        break

    print("part 2")
    faces = face_cascade.detectMultiScale(new_img, 1.35, 5)
    roi_gray = new_img
    # iteriramo kroz sve detekcije korak 1.
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        # filtering small imange if it gets caught first
        if w < 150:
            continue
        cv2.rectangle(new_img , (x - 200, y + h + 100), (x + 1700, y + h + 500), (255, 255, 0), 2)
        hmax, wmax = new_img.shape[:2]
        h_max_crr = y + h + 500
        w_max_crr = x + 1700

        print(hmax, wmax)
        print(h_max_crr, w_max_crr)

        if h_max_crr > hmax:
            h_max_crr = h_max_crr
        if w_max_crr > wmax:
            w_max_crr = wmax
        wmin = x-200
        if wmin < 0:
            wmin = 0
        roi_gray = new_img[y + h + 100:h_max_crr, wmin:w_max_crr]


    # find biggest conture
    # find angle
    # rotate
    # repeat

    #   plt.imshow(roi_gray, 'gray')
    #   plt.imshow(processed_img, 'gray')
    #   plt.imshow(img, 'gray')
    #   plt.show()

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    # odaberemo Tessract - prvi na listi ako je jedini alat
    tool = tools[0]
    print("Koristimo backend: %s" % (tool.get_name()))
    # biramo jezik očekivanog teksta
    lang = 'eng'

    text = tool.image_to_string(
        Image.fromarray(roi_gray),
        lang=lang,
        builder=pyocr.builders.TextBuilder(tesseract_layout=3)  # izbor segmentacije (PSM)
    )

    person.fit(text)
    print(person.expiry_date)
    print(person.birth_date)
    print(person.name)

    return person

def test(image_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # ucitavanje i transformacija slike
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    screen = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detekcija svih lica na grayscale slici
    faces = face_cascade.detectMultiScale(screen, 1.35, 5)
    # iteriramo kroz sve detekcije korak 1.
    for (x, y, w, h) in faces:

        cv2.rectangle(screen, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = screen[y:y + h, x:x + w]
        roi_color = screen[y:y + h, x:x + w]
        bin = get_binary_image(roi_gray)

        eyes = eye_cascade.detectMultiScale(roi_gray)

        # finding the largest pair of
        # eyes in the image
        if len(eyes) >= 2:
            eye = eyes[:, 2]
            container1 = []
            for i in range(0, len(eye)):
                container = (eye[i], i)
                container1.append(container)
            df = pd.DataFrame(container1, columns=[
                "length", "idx"]).sort_values(by=['length'])
            eyes = eyes[df.idx.values[0:2]]


        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)
        break

    plt.imshow(screen, 'gray')
    plt.show()

def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 30, 255, cv2.THRESH_BINARY_INV)
    return image_bin


def get_binary_image(img_base):
    #image_ada = cv2.GaussianBlur(image_ada, (11, 11), 5)
    image_ada_bin = cv2.adaptiveThreshold(img_base, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY_INV, 251, 1)
    return image_ada_bin

def invert(image):
    return 255-image


# Detect face
def face_detection(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.35, 5)
    faces.sort()
    if len(faces) <= 0:
        return img, img
    else:
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            img2 = img[y:y + h, x:x + w]
            return img, img2


def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
                    ((b[1] - a[1]) * (b[1] - a[1])))


# Find eyes
def face_alignment(img_path):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img_raw = cv2.imread(img_path)
    #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    screen = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img, gray_img = face_detection(screen)
    plt.imshow(img, 'gray')
    plt.show()
    eyes = eye_cascade.detectMultiScale(gray_img)

    # for multiple people in an image find the largest
    # pair of eyes
    new_img = img
    if len(eyes) >= 2:
        eye = eyes[:, 2]
        container1 = []
        for i in range(0, len(eye)):
            container = (eye[i], i)
            container1.append(container)
        df = pd.DataFrame(container1, columns=[
                        "length", "idx"]).sort_values(by=['length'])
        eyes = eyes[df.idx.values[0:2]]

        # deciding to choose left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] > eye_2[0]:
            left_eye = eye_2
            right_eye = eye_1
        else:
            left_eye = eye_1
            right_eye = eye_2

        # center of eyes
        # center of right eye
        right_eye_center = (
            int(right_eye[0] + (right_eye[2]/2)),
        int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

        # center of left eye
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
        int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]
        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

        # finding rotation direction
        if left_eye_y > right_eye_y:
            print("Rotate image to clock direction")
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 # rotate image direction to clock
        else:
            print("Rotate to inverse clock direction")
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 # rotate inverse direction of clock

        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
        a = trignometry_for_distance(left_eye_center,
                                    point_3rd)
        b = trignometry_for_distance(right_eye_center,
                                    point_3rd)
        c = trignometry_for_distance(right_eye_center,
                                    left_eye_center)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = (np.arccos(cos_a) * 180) / math.pi

        if direction == -1:
            angle = 90 - angle
        else:
            angle = -(90-angle)

        # rotate image
        new_img = img
        h, w = new_img.shape[:2]
        new_img = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        print("asdadsasdads")
        print(angle)
    return new_img
