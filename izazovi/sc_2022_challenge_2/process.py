import copy
import os

import sklearn.linear_model

os.environ['KERAS_BACKEND'] = 'theano'

from fuzzywuzzy import fuzz
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from sklearn.cluster import KMeans

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
    resized = cv2.resize(region,(32,32), interpolation = cv2.INTER_NEAREST)
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
    ann.add(Dense(500, input_dim=1024, activation='sigmoid'))
    ann.add(Dense(200, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=20000, batch_size=1, verbose=0, shuffle=True)

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
    #TODO fixes crashes on bad regions
    # regions_array.remove(region2) is creating problems
    good_regions = copy.deepcopy(regions_array)
    bad_regions = []
    i = 0
    for region in regions_array:
        for region2 in regions_array:
            # x11 <  x21 and x21 < x12   ## x2 is contained in x1
            if region is not region2 and region[1][0] < region2[1][0] < region[1][0] + region[1][2] and region2[1][0] + abs((region2[1][0] - abs(region2[1][0] - region2[1][2])))/2 < region[1][0] + region[1][2]:
                # height = height1 + height2 + difference between contures
                # y1 = y2 - moving y point up
                if region2[1][1] < region[1][1]:
                    prvi = region
                    drugi = region2
                else:
                    prvi = region2
                    drugi = region

                region[1] = (prvi[1][0], drugi[1][1], prvi[1][2],
                             prvi[1][3] + np.abs(prvi[1][1] - drugi[1][1] - drugi[1][3]) + drugi[1][3])
                good_regions[i][1] = (prvi[1][0], prvi[1][1], prvi[1][2], prvi[1][3])

        i += 1

    average_size = 0
    for gr in good_regions:
        average_size += gr[1][2] * gr[1][3]
    average_size = average_size/len(good_regions)

    i = 0
    br = []
    for gr in good_regions:
        if gr[1][2] * gr[1][3] < average_size/5:
            br.append(i)
        i += 1
    br = list(set(br))
    for b in sorted(br, reverse=True):
        del good_regions[b]

    i = 0
    br = []
    for i in range(0, len(good_regions)-1):
        for j in range(i+1, len(good_regions)):
            if good_regions[i][1][0] == good_regions[j][1][0]:
                br.append(i)

    br = list(set(br))
    print(br)
    for b in sorted(br, reverse=True):
        del good_regions[b]

    return good_regions

def select_roi(image_bin,img_base):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []

    contours2 = []
    xs = []
    ys = []
    for contour in contours:
        if len(contour) < 10:
            continue
        contours2.append(contour)
        x, y, w, h = cv2.boundingRect(contour)
        xs.append(x+w/2)
        ys.append(y+h/2)
    if len(xs) == 0 or len(ys) == 0:
        return img_base, [], []

    reg = sklearn.linear_model.LinearRegression().fit(np.array(xs).reshape(-1, 1), np.array(ys))
    first = (0, reg.predict(np.array([0]).reshape(-1,1)))
    second = (100, reg.predict(np.array([100]).reshape(-1,1)))
    arctan_num = abs(first[1][0]-second[1][0])/abs(first[0]-second[0])
    angle = np.arctan(arctan_num)*180/np.pi
    if first[1][0] > second[1][0]:
        angle *= -1
    h,w = image_bin.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2) , angle, 1.0)
    image_bin2 = cv2.warpAffine(image_bin, m, (w, h))

    show_image(image_bin2)
    contours = contours2
    i = 0
    for contour in contours:
        if len(contour) < 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1, x:x+w+1]
        regions_array.append([region, (x-1, y-1, w+2, h+2)])

    # rotating picture based on the axis of the text

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    # combining regions
    regions_array = combine_regions(regions_array)
    regions_array = [[resize_region(reg[0]), (reg[1][0], reg[1][1], reg[1][2], reg[1][3])] for reg in regions_array]

    #for r in regions_array:
        #x,y,w,h = r[1]
        #cv2.rectangle(img_base, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #show_image(img_base)
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

    return img_base, sorted_regions, region_distances

def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta (obe fotografije)
    :return: Objekat modela
    """
    # probaj da ucitas prethodno istreniran model
    model = load_trained_ann()
    #model = None
    if model is not None:
        return model

    imgs = []
    imgs2 = []
    for i in range(0, len(train_image_paths)):
        img = load_image(train_image_paths[i])
        imgs2.append(img)
        imgs.append(invert(image_bin(image_gray(img))))

    if 'alphabet0.bmp' not in train_image_paths[0]:
        swap = imgs[0]
        imgs[0] = imgs[1]
        imgs[1] = swap[0]

        swap2 = imgs2[0]
        imgs2[0] = imgs2[1]
        imgs2[0] = swap2

    img1, letters1, region_distances1 = select_roi(imgs[0], imgs2[0])
    img2, letters2, region_distances2 = select_roi(imgs[1], imgs2[1])

    show_image(img1)

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

def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result

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

    img_base = cv2.imread(image_path)
    img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    # fine-tuning practically
    ranged = 19
    lista = get_most_prominent_colors(img_base, ranged)
    print(lista)
    print(image_path)
    # TODO FUZZY WUZZY
    # TODO RELATIVE SIZE CLEANSING
    # TODO
    for i in range(0, len(lista)):
        img = create_bin_image_based_on_color(img_base, lista[i][0], ranged)

        img = img_to_binary(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

        #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        selected_regions, letters, distances = select_roi(img,img_base)
        if 15 < len(letters) < 150:
            break

    print('Broj prepoznatih regiona:', len(letters))

    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente

    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(distances)

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž',
                'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    extracted_text = (display_result(results, alphabet, k_means))


    extracted_text_list = extracted_text.split(" ")
    all_values = 0
    for v in vocabulary.values():
        all_values += int(v)
    ll = []
    for text in extracted_text_list:
        best_value = 0
        best_match = ''
        for v in vocabulary.items():
            temp = fuzz.ratio(text, v[0])
            if best_value < temp/all_values:
                best_value = temp/all_values
                best_match = v[0]
        ll.append(best_match)

    extracted_text2 = ''
    for t in ll:
        extracted_text2 += t + ' '

    print(extracted_text)
    print(extracted_text2)
    #show_image(img)
    return extracted_text2


def get_most_prominent_colors(img_base, ranged):
    dict1 = {}
    for j in range(0,len(img_base)):
        for i in img_base[j]:
            tuplei = (i[0], i[1], i[2])
            if tuplei in dict1:
                dict1[tuplei] += 1
            else:
                dict1.update({tuplei: 1})

    gray_noise_removed = remove_noise_from_color_list(dict1, 3)
    sorted_list = sorted(gray_noise_removed.items(), key=lambda x: x[1], reverse=False)
    sorted_dict = {}
    for item in sorted_list:
        sorted_dict.update({item[0]: item[1]})

    list_to_return = []
    for m in range(0, 200):
        list_to_return.append(sorted_dict.popitem())
    sorted_dict = {}
    for item in list_to_return:
        sorted_dict.update({item[0]: item[1]})

    color_noise_removed = remove_noise_from_text(sorted_dict, ranged)
    sorted_dict = {}
    for item in list_to_return:
        sorted_dict.update({item[0]: item[1]})
    sorted_list = sorted(sorted_dict.items(), key=lambda x: x[1], reverse=False)
    sorted_dict = {}
    for item in sorted_list:
        sorted_dict.update({item[0]: item[1]})
    list_to_return = []
    for m in range(0, 20):
        list_to_return.append(sorted_dict.popitem())

    return color_noise_removed


def remove_noise_from_text(color_dict, rangeb):
    list_done = []
    for item in color_dict.items():
        found = False
        if len(list_done) == 0:
            list_done.append(item)
        for i in range(0, len(list_done)):
            if rangeb + list_done[i][0][0] > item[0][0] > list_done[i][0][0] - rangeb and rangeb + list_done[i][0][1] \
                    > item[0][1] > list_done[i][0][1] - rangeb and rangeb + list_done[i][0][2] > item[0][2] > list_done[i][0][2] - rangeb:
                list_done[i] = ((list_done[i][0][0], list_done[i][0][1], list_done[i][0][2]), item[1]+list_done[i][1])
                found = True
                break

        if found is False:
            list_done.append(item)
    return list_done


def remove_noise_from_color_list(color_list, split_number):
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
            a = 0
            #color_list[(item[2], item[2], item[2])] += item[1]
        else:
            a = 0
            #color_list.update({(item[2], item[2], item[2]): item[1]})

    return color_list


def change_one_color_to_another(img, color1, color2):
    color = np.array(color1)
    mask = cv2.inRange(img, color1, color2)
    result = cv2.bitwise_and(img, img, mask=mask)
    img_binary = cv2.bitwise_and(img, result)
    return img


def create_bin_image_based_on_color(img, color, ranged):
    color1 = np.array((color[0]-ranged, color[1]-ranged, color[2]-ranged))
    color2 = np.array((color[0]+ranged, color[1]+ranged, color[2]+ranged))
    mask = cv2.inRange(img, color1, color2)
    result = cv2.bitwise_and(img, img, mask=mask)
    img_binary = cv2.bitwise_and(img, result)
    return img_binary


def img_to_binary(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    return img_bin


def get_binary_image(img_base):
    image_ada = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    #image_ada = cv2.GaussianBlur(image_ada, (11, 11), 5)
    image_ada_bin = cv2.adaptiveThreshold(image_ada, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY_INV, 51, 5)
    return image_ada_bin


def show_image(img):
    plt.imshow(img)
    #plt.show()
    return
