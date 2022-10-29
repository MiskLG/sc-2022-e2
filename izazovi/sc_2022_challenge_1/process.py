# ovde importovati biblioteke
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca.

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca
    """
    blood_cell_count = 0
    # TODO - Prebrojati crvena krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    img_base = cv.imread(image_path)
    img_base = cv.cvtColor(img_base, cv.COLOR_BGR2RGB)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharpened = cv.filter2D(src=img_base, ddepth=-1, kernel=kernel)
    #print(kernel)
    # img_sharpened = cv.filter2D(img_base,0, kernel)

    #show_image(img_sharpened)

    img_binary = get_binary_image(img_base)
    print(image_path)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_binary = cv.erode(img_binary, kernel, iterations=3)
    img_binary = cv.dilate(img_binary, kernel, iterations=6)


    img, contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    img = img_base.copy()


    contours_updated = 0
    print(len(contours))


    #contours_map = [len(con) > 50 for con in contours]

    popup = []
    for i in range(0,len(contours)):
        if len(contours[i]) < 200 or len(contours[i]) > 600:
            popup.append(i)
    for i in popup[::-1]:
        contours.pop(i)

    #print(len(contours_map))
    #print(contours)
    #print(contours_map)
    #contours_updated = contours[contours_map]

    print(contours_updated)
    #print(len(contours_updated))
    #contours_updated = np.array(contours_updated)
    #cv.drawContours(img, contours, -1, (255, 0, 0), 1)
    img_binary = 0*img_binary
    cv.drawContours(img_binary, contours, -1, (151, 113, 222), cv.FILLED)


    kernel = np.ones((3, 3), np.uint8)

    # img_binary = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=2)
    # img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, iterations=1)
    # sure background area

    dist_transform = cv.distanceTransform(img_binary, cv.DIST_C, 3)
    ret, img_binary = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    img_binary = np.uint8(img_binary)
    img, contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    popup = []
    for i in range(0, len(contours)):
        if len(contours[i]) < 50 or len(contours[i]) > 600:
            popup.append(i)
    for i in popup[::-1]:
        contours.pop(i)

    show_image(img_binary)

    for con in contours:
        centers, radius = cv.minEnclosingCircle(con)
        print(centers)
        centers = tuple([(int(element[0]), int(element[1])) for element in [centers]])
        print(centers[0])
        radius = int(np.floor(radius))
        cv.circle(img_binary, centers[0], radius, (151, 113, 222), cv.FILLED)

    print(centers)
    #print(contours)
    print(contours_updated)
    #print(len(contours_updated))
    print(len(contours))
    show_image(img_binary)

    #cv.fillPoly(img_binary, pts=[contours], color=(255, 255, 255))

    blood_cell_count = len(contours)
    #print(centers)
    # TODO lower size, watershed
    '''
    #opening = cv.morphologyEx(img_ero, cv.MORPH_OPEN, kernel, iterations=2)
    
     
    
    sure_bg = cv.dilate(img_binary, kernel, iterations=2)

    dist_transform = cv.distanceTransform(img_binary, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    ret, markers = cv.connectedComponents(img_binary)
    markers = markers + 1
    markers = cv.watershed(img_base, markers)
    '''



    # noise removal
    kernel = np.ones((3, 3), np.uint8)

    #img_binary = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=2)
    #img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, iterations=1)
    # sure background area
    sure_bg = cv.dilate(img_binary, kernel, iterations=3)

    # Finding sure foreground area

    #img, contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#    centers, radius = cv.minEnclosingCircle(contours[0])

 #   print(centers)
    dist_transform = cv.distanceTransform(img_binary, cv.DIST_L2, 3)
    ret, sure_fg = cv.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    #sure_fg = cv.erode(img_binary, kernel, iterations=20)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)


    unknown = cv.subtract(sure_bg, sure_fg)
    show_image(sure_fg)

    img, contours, hierarchy = cv.findContours(sure_fg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img_base, markers)
    img_base[markers == -1] = [255, 0, 0]
    print(len(markers))
    blood_cell_count = len(contours)
    #img, contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #img = img_base.copy()
    #print(markers == -1)
    #cv.drawContours(img, contours, -1, (151, 113, 222), 1,)

    #TODO cleanse blue parts

    #for con in contours:


    #TODO find center of the counture , increase its radius untill it reaces the end of the countiure

    #TODO make those contures as primary targets and cleanse them from big ones and small ones.


    #plt.imshow(img_tr, 'gray')
    #plt.imshow(sure_fg)
    #plt.show()
    #blood_cell_count = len(contours)
    return blood_cell_count


def get_binary_image(img_base):
    gray = cv.cvtColor(img_base, cv.COLOR_RGB2GRAY)
    ret, img_tr = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # img_tr = gray > 180
    image_ada = cv.cvtColor(img_base, cv.COLOR_RGB2GRAY)

    ret, image_ada_bin = cv.threshold(image_ada, 100, 255, cv.THRESH_BINARY)

    image_ada_bin = cv.adaptiveThreshold(image_ada, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 151, 2)

    return image_ada_bin


def show_image(img):
    plt.imshow(img)
    #plt.show()
    return


