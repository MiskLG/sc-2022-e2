import os
os.environ['KERAS_BACKEND'] = 'theano'

import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 190, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose=0, shuffle=False)

    return ann


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialization_folder/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None


def combine_regions(regions_array):
    for region in regions_array:
        for region2 in regions_array:
            # x11 <  x21 and x21 < x12   ## x2 is contained in x1
            if region[1][0] < region2[1][0] < region[1][0] + region[1][2] and region2[1][0] + abs((region2[1][0] - abs(region2[1][0] - region2[1][2])))/2 < region[1][0] + region[1][2]:
                # height = height1 + height2 + difference between contures
                # y1 = y2 - moving y point up
                region[1] = (region[1][0], region2[1][1], region[1][2], region[1][3] + np.abs(region[1][1]-region2[1][1]) + region2[1][3])
                regions_array.remove(region2)

    return regions_array

def select_roi(image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1, x:x+w+1]
        regions_array.append([region, (x, y, w, h)])
    # combining regions
    regions_array = combine_regions(regions_array)
    regions_array = [[resize_region(reg[0]), (reg[1][0], reg[1][1], reg[1][2], reg[1][3])] for reg in regions_array]
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return sorted_regions, region_distances

def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta (obe fotografije)
    :return: Objekat modela
    """
    # probaj da ucitas prethodno istreniran model
    return None
    model = load_trained_ann()
    if model is not None:
        return model

    imgs = []
    for i in range(0, len(train_image_paths)):
        img = load_image(train_image_paths[i])
        imgs.append(invert(image_bin(image_gray(img))))

    letters1, region_distances1 = select_roi(imgs[0])
    letters2, region_distances2 = select_roi(imgs[1])

    print(len(letters1))
    print(len(letters2))

    for let in letters2:
        letters1.append(let)

    letters = letters1

    print(len(letters))

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž',
                'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    inputs = prepare_for_ann(letters)
    outputs = convert_output(alphabet)


    print('Broj prepoznatih regiona:', len(letters))

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    print("Traniranje modela zapoceto.")
    model = create_ann()
    model = train_ann(model, inputs, outputs)
    print("Treniranje modela zavrseno.")
    # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
    serialize_ann(model)

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
    extracted_text = "a"
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    img_base = cv2.imread(image_path)
    img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    lista = get_most_prominent_colors(img_base)
    print(lista)
    print(image_path)

    img = create_bin_image_based_on_color(img_base, lista[2][0], 45)
    show_image(img)
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


    #sorted_dict2.update({item[0]: item[1]} for item in sorted_dict1)
    gray_noise_removed = remove_noise_from_color_list(dict1, 3,img_base)
    sorted_list = sorted(gray_noise_removed.items(), key=lambda x: x[1], reverse=False)
    sorted_dict = {}
    for item in sorted_list:
        sorted_dict.update({item[0]: item[1]})
    list_to_return = []
    for m in range(0,20):
        list_to_return.append(sorted_dict.popitem())
    return list_to_return


def remove_noise_from_text(color_dict, rangeb):
    list_to_do = []
    list_done = []
    for item in color_dict.items():
        if len(list_done) == 0:
            list_done.append(item)
        for i in range(0, len(list_done)):
            if rangeb + list_done[i][0][0] > item[0][0] > list_done[i][0][0] - rangeb or rangeb + list_done[i][0][1] \
                    > item[0][1] > list_done[i][0][1] - rangeb or rangeb + list_done[i][0][2] > item[0][2] > list_done[i][0][2] - rangeb:
                list_to_do.append((item[0], item[1], (list_done[i][0][1], list_done[i][0][1], list_done[i][0][2]) ))

    for item in list_to_do:
        color_dict.pop(item[0])
        if (item[2], item[2], item[2]) in color_dict:
            color_dict[(item[2], item[2], item[2])] += item[1]
        else:
            color_dict.update({(item[2][0], item[2][1], item[2][2]): item[1]})
    return color_dict


def remove_noise_from_color_list(color_list, split_number, img):
    step = 255/split_number
    step_list = []
    for i in range(0, split_number+1):
        step_list.append(round(255-i*step))
    print(step_list)
    list_to_do = []
    for item in color_list.items():
        if item[0][0] == item[0][1] == item[0][2]:
            for i in range(0, split_number+1):
                if 1 + round(step / 2) + step_list[i] > item[0][0] > step_list[i] - round(step / 2) - 1:
                    list_to_do.append((item[0], item[1], step_list[i]))

    for item in list_to_do:
        color_list.pop(item[0])
        if (item[2], item[2], item[2]) in color_list:
            color_list[(item[2], item[2], item[2])] += item[1]
        else:
            color_list.update({(item[2], item[2], item[2]): item[1]})

    return color_list


def change_one_color_to_another(img, color1, color2):
    color = np.array(color1)
    mask = cv2.inRange(img, color1, color2)
    result = cv2.bitwise_and(img, img, mask=mask)
    img_binary = cv2.bitwise_and(img, result)
    return img


def create_bin_image_based_on_color(img, color1, ranged):
    color1 = np.array((color1[0]-ranged, color1[1]-ranged, color1[2]-ranged))
    color2 = np.array((color1[0]+ranged, color1[1]+ranged, color1[2]+ranged))
    mask = cv2.inRange(img, color1, color2)
    result = cv2.bitwise_and(img, img, mask=mask)
    img_binary = cv2.bitwise_and(img, result)
    return img_binary

def get_binary_image(img_base):
    image_ada = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    #image_ada = cv2.GaussianBlur(image_ada, (11, 11), 5)
    image_ada_bin = cv2.adaptiveThreshold(image_ada, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY_INV, 51, 5)
    return image_ada_bin


def show_image(img):
    plt.imshow(img)
    plt.show()
    return
