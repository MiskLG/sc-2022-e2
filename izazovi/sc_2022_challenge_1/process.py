# ovde importovati biblioteke
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def count_blood_cells2(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca.

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca
    """
    blood_cell_count = 0
    # TODO - Prebrojati crvena krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    img_base = cv.imread(image_path)
    img_base = cv.cvtColor(img_base, cv.COLOR_BGR2HSV)


    lower_range = np.array([150, 0, 0])
    upper_range = np.array([200, 500, 500])

    mask = cv.inRange(img_base, lower_range, upper_range)  # Create a mask with range
    result = cv.bitwise_and(img_base, img_base, mask=mask)
    show_image(mask)

    show_image(cv.cvtColor(result, cv.COLOR_HSV2RGB))

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharpened = cv.filter2D(src=img_base, ddepth=-1, kernel=kernel)
    #print(kernel)
    # img_sharpened = cv.filter2D(img_base,0, kernel)

    #show_image(img_sharpened)

    img_binary = get_binary_image(result)
    print(image_path)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, iterations=1)




    #img_binary = cv.erode(img_binary, kernel, iterations=3)

    #img, contours, hierarchy = cv.findContours(img_binary, cv., cv.CHAIN_APPROX_NONE)
    #cv.drawContours(img_binary, contours, -1, (151, 113, 222), cv.FILLED)
    #show_image(img_binary)



    #dist_transform = cv.distanceTransform(img_binary, cv.DIST_C, 3)
    #ret, img_binary = cv.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    img_binary = cv.cvtColor(img_binary, cv.COLOR_BGR2GRAY)
    img, contours, hierarchy = cv.findContours(img_binary, cv.CV_8UC1, cv.CHAIN_APPROX_NONE)
    img = img_base.copy()


    contours_updated = 0
    print(len(contours))
    cv.drawContours(img_binary, contours, -1, (151, 113, 222), 1)
    show_image(img_binary)
    #contours_map = [len(con) > 50 for con in contours]
    #img_binary = cv.erode(img_binary, kernel, iterations=9)

    popup = []
    for i in range(0,len(contours)):
        if len(contours[i]) < 100:
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
    cv.drawContours(img_binary, contours, -1, (151, 113, 222), cv.CHAIN_APPROX_SIMPLE)


    kernel = np.ones((3, 3), np.uint8)

    # img_binary = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=2)
    # img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, iterations=1)
    # sure background area


    #dist_transform = cv.distanceTransform(img_binary, cv.DIST_L2, 3)
    #ret, img_binary = cv.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    #img_binary = np.uint8(img_binary)
    #img, contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)



    popup = []
    for i in range(0, len(contours)):
        if len(contours[i]) < 20:
            popup.append(i)
    for i in popup[::-1]:
        contours.pop(i)

    img = img_base.copy()
    cv.drawContours(img, contours, -1, (151, 113, 222), 1)
    show_image(img)

    img_binary = img_binary * 0
    for i in range(0, len(contours)):
        centers, radius = cv.minEnclosingCircle(contours[i])
        print(centers)
        centers = tuple([(int(element[0]), int(element[1])) for element in [centers]])
        print(centers[0])

        radius = int(np.floor(radius))
        if not (centers[0][0] < 25 or centers[0][0] > 615 or centers[0][1] < 25 or centers[0][1] > 455):
            print("asdadsadadsasdsadasddasdsads")
            print(cv.contourArea(contours[i]))
            print(radius*radius*np.pi)
            cv.circle(img_binary, centers[0], 40, (151, 113, 222), cv.FILLED)

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
    ret, sure_fg = cv.threshold(dist_transform, 0.23 * dist_transform.max(), 255, 0)
    #sure_fg = cv.erode(img_binary, kernel, iterations=20)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    img, contours, hierarchy = cv.findContours(sure_fg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    print(len(markers))
    print(ret)
    markers = cv.watershed(img_base, markers)

    img_base[markers == -1] = [255, 0, 0]
    print(len(markers))
    blood_cell_count = ret
    #img, contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #img = img_base.copy()
    #print(markers == -1)
    #cv.drawContours(img, contours, -1, (151, 113, 222), 1,)

    markers = np.zeros(sure_fg.shape, dtype=np.int32)
    dist_8u = sure_fg.astype('uint8')
    img, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        img = cv.drawContours(markers, contours, i, (i + 1), -1)

    show_image(img)
    #TODO cleanse blue parts

    #for con in contours:


    #TODO find center of the counture , increase its radius untill it reaces the end of the countiure

    #TODO make those contures as primary targets and cleanse them from big ones and small ones.

    #TODO remove purple blobs, remove uncertain cellz
    #plt.imshow(img_tr, 'gray')
    #plt.imshow(sure_fg)
    #plt.show()
    #blood_cell_count = len(contours)
    return blood_cell_count

def count_blood_cells(image_path):
    blood_cell_count = 0
    # TODO - Prebrojati crvena krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    # Setup for image processing
    img_base = cv.imread(image_path)
    img_base = cv.cvtColor(img_base, cv.COLOR_BGR2RGB)
    print(image_path)

    show_image(img_base)
    # Get binary image using gaussian adaptive method
    img_binary = get_binary_image2(img_base)
    #show_image(img_binary)

    # Remove contours that contain blue tint
    lower_range = np.array([60, 35, 150])
    upper_range = np.array([180, 255, 255])

    img_hsv = cv.cvtColor(img_base, cv.COLOR_RGB2HSV)
    mask = cv.inRange(img_hsv, lower_range, upper_range)  # Create a mask with range

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    show_image(mask)

    result = cv.bitwise_and(img_binary, img_binary, mask=mask)
    img_binary = cv.bitwise_xor(img_binary, result)
    show_image(result)

    # Remove small noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=1)
    show_image(img_binary)

    # Connect contours
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    #img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, iterations=2)

    # Remove small contours
    img, contours, hierarchy = cv.findContours(img_binary, cv.CV_8UC1, cv.CHAIN_APPROX_SIMPLE)
    popup = []
    for i in range(0, len(contours)):
        if len(contours[i]) < 80:
            popup.append(i)
    for i in popup[::-1]:
        contours.pop(i)

    # Clear image and draw contours
    img_binary = 0 * img_binary
    cv.drawContours(img_binary, contours, -1, (255, 255, 255), cv.FILLED)
    show_image(img_binary)

    img_binary = 0 * img_binary
    for i in range(0, len(contours)):
        centers, radius = cv.minEnclosingCircle(contours[i])
        centers = tuple([(int(element[0]), int(element[1])) for element in [centers]])

        radius = int(np.floor(radius))
        if not (centers[0][0] < 30 or centers[0][0] > 610 or centers[0][1] < 30 or centers[0][1] > 450):
            cv.circle(img_binary, centers[0], 40, (151, 113, 222), cv.FILLED)
    show_image(img_binary)

    # WATERSHED?
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel)
    show_image(img_binary)

    img, contours, hierarchy = cv.findContours(img_binary, cv.CV_8UC1, cv.CHAIN_APPROX_SIMPLE)

    return len(contours)

def get_binary_image2(img_base):
    image_ada = cv.cvtColor(img_base, cv.COLOR_RGB2GRAY)
    image_ada = cv.GaussianBlur(image_ada, (7, 7), 1)
    image_ada_bin = cv.adaptiveThreshold(image_ada, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 101, 1)
    return image_ada_bin

def get_binary_image(img_base):
    image_ada = cv.cvtColor(img_base, cv.COLOR_HSV2RGB)

    ret, image_ada_bin = cv.threshold(image_ada, 1, 255, cv.THRESH_BINARY)
    #image_ada_bin = cv.adaptiveThreshold(image_ada, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 151, 2)
    return image_ada_bin


def show_image(img):
    plt.imshow(img)
    #plt.show()
    return


