# import libraries here

import sys

import cv2
import dlib
import matplotlib.pyplot as plt
import pyocr
import pyocr.builders
from PIL import Image
from imutils import face_utils


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """

    def __init__(self, birth_date: str, expiry_date: str, gender: str, name: str, surname: str, number: str):
        self.birth_date = birth_date
        self.expiry_date = expiry_date
        self.gender = gender
        self.name = name
        self.surname = surname
        self.number = number

    def fit(self, text):
        # TODO add filtering for single text and return best thingy
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
                y = t.find('<', x)
                if x == -1 or y == -1:
                    break
                self.surname = t[x + 3:y]
                z = t.find('<', y + 2)
                if z == -1:
                    break
                self.name = t[y + 2: z]
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
                temp = t[x + 3:x + 3 + 6]
                self.birth_date = self.format_date(temp, "19")

                temp = t[y + 1: y + 1 + 6]
                self.expiry_date = self.format_date(temp, "20")
                break
        return self

    def fit2(self, text):
        #TODO number-text converter
        #TODO more variants for SRB, 5R8
        #TODO tidyup dates
        #TODO tidyup < char and empty spaces
        value = 0
        first_line = text.find('SRB')
        if first_line == -1:
            first_line = text.find('8RB')
            if first_line == -1:
                first_line = text.find('SR8')
                if first_line == -1:
                    first_line = text.find('5RB')
                    if first_line == -1:
                        first_line = text.find('PRB')
                        if first_line == -1:
                            return value
        # can extract name and last name probably
        test3 = False
        value += 1  # value = 1
        second_line = text.find('SRB', first_line + 3)
        if second_line == -1:
            second_line = text.find('8RB', first_line + 3)
            if second_line == -1:
                second_line = text.find('SR8', first_line + 3)
                if second_line == -1:
                    second_line = text.find('5RB', first_line + 3)
                    if second_line == -1:
                        second_line = text.find('PRB', first_line + 3)
                        if second_line == -1:
                            test3 = True
        # can extract data somehow

        if test3:
            second_line = text.find('SRB')
            if second_line == -1 or second_line == first_line:
                second_line = text.find('8RB')
                if second_line == -1 or second_line == first_line:
                    second_line = text.find('SR8')
                    if second_line == -1 or second_line == first_line:
                        second_line = text.find('5RB')
                        if second_line == -1 or second_line == first_line:
                            second_line = text.find('PRB')
                            if second_line == -1 or second_line == first_line:
                                return value
            if first_line > second_line:
                second_line, first_line = first_line, second_line
        value += 1  # value = 2
        print("passed")

        surname_end = text.find('<', first_line + 3)
        if surname_end != -1:
            value += 1  # value = 3
        self.surname = text[first_line + 3:surname_end]
        if len(self.surname) < 16:
            value += 1  # v = 4
        self.surname = self.number_text_converter(self.surname, 0)
        name_end = text.find('<', surname_end + 2)  # usually there should be 2
        if name_end != -1:
            value += 1
        self.name = text[surname_end + 2: name_end]
        if len(self.name) < 12:
            value += 1  # value = 5
        self.name = self.number_text_converter(self.name, 0)
        crr = second_line - 10
        if crr < 0:
            crr = 0
        self.number = text[crr:second_line]
        if self.number.isdigit():
            value += 1  # value = 6
        self.number = self.number_text_converter(self.number, 1)


        gender = text.find('F', second_line + 3)
        if gender == -1:
            self.gender = 'M'
            gender = text.find('M', second_line + 3)
            if gender != -1:
                value += 1  # v =7
        else:
            self.gender = 'F'
            value += 1  # value = 7

        temp = text[second_line + 3:second_line + 3 + 6]
        self.birth_date = self.format_date(temp, "19")
        self.birth_date = self.number_text_converter(self.birth_date, 1)
        if temp.isdigit():
            value += 1  # value = 8

        temp = text[gender + 1: gender + 1 + 6]
        self.expiry_date = self.format_date(temp, "20")
        self.expiry_date = self.number_text_converter(self.expiry_date, 1)
        if temp.isdigit():
            value += 1  # value = 9
        return value

    def format_date(self, date, type):
        year = str("" + type + date[:2])
        month = date[2:4]
        day = date[4:6]
        return day + "." + month + "." + year

    '''
        convert_type: 0 for number to text
                      1 for text to number
    '''
    def number_text_converter(self, text, convert_type):
        mix_file = []
        mix_file.append(('4', 'A'))
        mix_file.append(('8', 'B'))
        mix_file.append(('0', 'C'))
        mix_file.append(('6', 'D'))
        mix_file.append(('3', 'E'))
        mix_file.append(('4', 'F'))
        mix_file.append(('6', 'G'))
        mix_file.append(('2', 'H'))
        mix_file.append(('1', 'I'))
        mix_file.append(('9', 'J'))
        mix_file.append(('7', 'K'))
        mix_file.append(('1', 'L'))
        mix_file.append(('2', 'M'))
        mix_file.append(('2', 'N'))
        mix_file.append(('0', 'O'))
        mix_file.append(('4', 'P'))
        mix_file.append(('8', 'R'))
        mix_file.append(('5', 'S'))
        mix_file.append(('1', 'T'))
        mix_file.append(('2', 'U'))
        mix_file.append(('7', 'V'))
        mix_file.append(('2', 'Z'))

        mix_file2 = []
        mix_file2.append(('0', '0'))
        mix_file2.append(('1', 'I'))
        mix_file2.append(('2', 'Z'))
        mix_file2.append(('3', 'E'))
        mix_file2.append(('4', 'A'))
        mix_file2.append(('5', 'S'))
        mix_file2.append(('6', 'G'))
        mix_file2.append(('7', 'V'))
        mix_file2.append(('8', 'B'))
        mix_file2.append(('9', 'P'))

        if convert_type == 0:
            for mix in mix_file2:
                text = text.replace(mix[0], mix[1])
        else:
            for mix in mix_file:
                text = text.replace(mix[1], mix[0])

        return text

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
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # odaberemo Tessract - prvi na listi ako je jedini alat
    tool = tools[0]
    print("Koristimo backend: %s" % (tool.get_name()))
    # biramo jezik očekivanog teksta
    lang = 'eng'

    detector = dlib.get_frontal_face_detector()

    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    # predictor = dlib.shape_predictor('haarcascade_frontalface_default.xml')
    print("Start")
    # test(image_path)

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    # odaberemo Tessract - prvi na listi ako je jedini alat

    # ucitavanje i transformacija slike
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    screen = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detekcija svih lica na grayscale slici
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    new_img = image

    show_img(image)
    (x, y, w, h) = (0, 0, 0, 0)
    # iteriramo kroz sve detekcije korak 1.
    for f in faces:
        (x, y, w, h) = face_utils.rect_to_bb(f)
        print(x, y, w, h)
        # filtering small imange if it gets caught first
        if w < 200:
            continue

    # no faces no program
    if x == 0 and y == 0 and w == 0 and h == 0:
        return person

    # cv2.rectangle(new_img, (x - 200, y + h + 100), (x + 1700, y + h + 500), (255, 255, 0), 2)
    hmax, wmax = new_img.shape[:2]
    h_max_crr = y + h + 500
    w_max_crr = x + 1700

    print(hmax, wmax)
    print(h_max_crr, w_max_crr)

    if h_max_crr > hmax:
        h_max_crr = h_max_crr
    if w_max_crr > wmax:
        w_max_crr = wmax
    wmin = x - 200
    if wmin < 0:
        wmin = 0
    roi_gray = new_img[y + h + 100:h_max_crr, wmin:w_max_crr]

    builder = pyocr.builders.WordBoxBuilder()
    builder.tesseract_flags = []
    builder.tesseract_flags.append(r"--psm")
    builder.tesseract_flags.append(r"11")

    builder.tesseract_flags.append(r"-c")
    builder.tesseract_flags.append(r"tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTUVZ<")

    text = tool.image_to_string(
        Image.fromarray(roi_gray),
        lang=lang,
        builder=builder
    )

    angle = -10
    ba = 100
    bt = 100
    bv = 0
    btt = 0
    for i in range(0, 40):
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        new_img = cv2.warpAffine(image, m, (w, h))
        faces = detector(new_img)
        x = 0
        y = 0
        w = 0
        h = 0
        for f in faces:
            (x2, y2, w2, h2) = face_utils.rect_to_bb(f)
            if w2 < 200:
                continue
            else:
                x = x2
                y = y2
                w = w2
                h = h2
                break

        # no faces no program
        if x == 0 and y == 0 and w == 0 and h == 0:
            continue
        hmax, wmax = new_img.shape[:2]
        h_max_crr = y + h + 500
        w_max_crr = x + 1700
        h_min_crr = y + h + 100

        if h_max_crr > hmax:
            h_max_crr = h_max_crr
        if w_max_crr > wmax:
            w_max_crr = wmax
        if h_min_crr > hmax:
            h_min_crr = hmax
        wmin = x - 200
        if wmin < 0:
            wmin = 0
        roi_gray = new_img[h_min_crr:h_max_crr, wmin:w_max_crr]
        angle += 0.5
        text = tool.image_to_string(
            Image.fromarray(roi_gray),
            lang=lang,
            builder=builder
        )
        if bt > len(text) >= 2:
            text_whole = ""
            for i in range(0, len(text)):
                text_whole += text[i].content
            ptemp = Person(None, None, None, None, None, None)
            value = ptemp.fit2(text_whole)
            print(text_whole)
            ba = angle
            bt = len(text)
            print(value)
            if bv <= value:
                bv = value
                person = ptemp
            if value == 10:
                break

    print(btt)
    print(ba)
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), ba, 1.0)
    new_img = cv2.warpAffine(image, m, (w, h))
    show_img(new_img)

    print(person.expiry_date)
    print(person.birth_date)
    print(person.name)

    return person


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 30, 255, cv2.THRESH_BINARY_INV)
    return image_bin


def get_binary_image(img_base):
    # image_ada = cv2.GaussianBlur(image_ada, (11, 11), 5)
    image_ada_bin = cv2.adaptiveThreshold(img_base, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY_INV, 251, 1)
    return image_ada_bin


def invert(image):
    return 255 - image


def show_img_gray(img):
    plt.imshow(img, 'gray')
    # plt.show()


def show_img(img):
    plt.imshow(img)
    # plt.show()
