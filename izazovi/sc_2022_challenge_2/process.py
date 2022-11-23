import cv2
from matplotlib import pyplot as plt
import numpy as np




def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta (obe fotografije)
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran
    model = None
    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    img_base = cv2.imread(image_path)
    img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    a = np.array(img_base)
    get_most_prominent_colors(img_base)
    print(a.shape)
    print(a)
    print(len(a))

    print(image_path)

    show_image(img_base)
    # Get binary image using gaussian adaptive method
    img_binary = get_binary_image(img_base)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)



    show_image(img_binary)

    return extracted_text

def get_most_prominent_colors(img_base):
    dict1 = {}
    print(len(img_base))
    for j in range(0,len(img_base)):
        for i in img_base[j]:
            tuplei = (i[0], i[1], i[2])
            if tuplei in dict1:
                dict1[tuplei] += 1
            else:
                dict1.update({tuplei: 1})

    sorted_dict1 = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
    for m in range(0,10):
        print(sorted_dict1[m])
    #print(dict1[(213, 193, 26)])
    return sorted_dict1


def get_binary_image(img_base):
    image_ada = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    #image_ada = cv2.GaussianBlur(image_ada, (11, 11), 5)
    image_ada_bin = cv2.adaptiveThreshold(image_ada, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY_INV, 51, 5)
    return image_ada_bin


def show_image(img):
    plt.imshow(img)
    plt.show()
    return
