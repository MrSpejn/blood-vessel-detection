from __future__ import division
from pylab import *
from skimage import data
from ipywidgets import *
import cv2.cv2
from PIL import Image
from copy import deepcopy
from sklearn.metrics import mean_squared_error

NUMBER_OF_IMAGES = 9
SHOW_EYE_CONTOUR = False
SMALLEST_ELEMENT_SIZE = 20
MASK_SIZE = 4
RESIZE_SIZE = int(876),int(584)
PICTURE_TYPE = "_g"

def load_image(file_location):
    image = data.load(file_location, as_grey=False)
    return image

def load_all_images_from_directory():
    image = []
    manually = []
    for i in range(NUMBER_OF_IMAGES):
        image.append(load_image("imagedatabase/0" + str(i+1) + PICTURE_TYPE + ".resized.jpg"))
        manually.append(load_image("imagedatabase/0" + str(i+1) + PICTURE_TYPE + "ref.resized.tif"))
    return image, manually


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def compare_each_in_area(image_orginal, image, x, y, mask):
    sum1 = 0
    sum0 = 0
    sum2 = 0
    d = 0
    for r in range(-1, 2, 2):
        for z in range(-1, 2, 2):
            for i in range(mask):
                for k in range(mask):
                    neighbour = image_orginal[x + r * i][y + z * k]
                    sum1 += neighbour[1]
                    sum0 += neighbour[0]
                    sum2 += neighbour[2]
                    d += 1
                    if not SHOW_EYE_CONTOUR and neighbour[0] == 0:
                        image[x][y] = [0, 0, 0]
                        break
    if 0.99*sum1/d < image[x][y][1] \
            or 0.99*sum0/d < image[x][y][0] or 0.99*sum2/d < image[x][y][2]:
        image[x][y] = [0, 0, 0]


def compare_neighbour_pixels(image_orginal, image, mask):
    for i in range(mask,len(image) - mask):
        for k in range(mask, len(image_orginal[2]) - mask):
            compare_each_in_area(image_orginal, image, i, k, mask)


def remove_small_parts(image, max_elem_size):
    c = 0
    im = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in im[1]:
        if len(i) < max_elem_size:
            for z in i:
                c += 1
                image[z[0][1]][z[0][0]] = 0
    return image


def convert_image(image, max_elem_size, mask):
    image_orginal = image.copy()
    compare_neighbour_pixels(image_orginal,image, mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for k in range(max_elem_size):
        image = remove_small_parts(image, max_elem_size)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image[image>1] = 255
    image[0:mask][:] = 0
    image[len(image) - mask:len(image)][:] = 0
    image[:][0:mask] = 0
    image[:][len(image) - mask:len(image)] = 0
    image[image<=1] = 0
    return image


def cover_image(image, contours):
    for i in range(len(image)):
        for k in range(len(image[0])):
            if contours[i][k] > 0:
                image[i][k] = [0, 0, 0]
            else:
                image[i][k] = [255, 255, 255]
    return image


def matrix_of_errors(image, manually):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(image)):
        for k in range(len(image[0])):
            if image[i][k] == manually[i][k] and image[i][k] > 0:
                tp += 1
                continue
            if image[i][k] == manually[i][k] and image[i][k] == 0:
                tn += 1
                continue
            if image[i][k] != manually[i][k] and image[i][k] > 0:
                fp += 1
                continue
            if image[i][k] != manually[i][k] and image[i][k] == 0:
                fn += 1
                continue
    return fp, fn, tp, tn

if __name__ == "__main__":   
    images, manually = load_all_images_from_directory()
    g_lpk = 0
    g_lbk = 0
    g_acc = 0
    g_err = 0
    g_czulosc = 0
    g_specyficznosc = 0
    g_ppv = 0
    g_npv = 0
    for i in range(len(images)):
        image = convert_image(deepcopy(images[i]), max_elem_size = SMALLEST_ELEMENT_SIZE, mask = MASK_SIZE)
        img = cover_image(images[i], image)
        display_image(img)
        # display_image(manually[i])
        fp, fn, tp, tn = matrix_of_errors(image, manually[i])
        p = tp + fn
        n = tn + fp
        lpk = tn + tp
        lbk = fp + fn
        acc = (tp + tn) / (p + n)
        err = (fp + fn) / (p + n)
        czulosc = tp / (tp + fn)
        specyficznosc = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        print("MSE= " + str(mean_squared_error(image, manually[i])))
        print("tp= " + str(tp))
        print("fn= " + str(fn))
        print("tn= " + str(tn))
        print("fp= " + str(fp))
        print("liczba poprawnych klasyfikacji = " + str(lpk))
        print("liczba blednych klasyfikacji = " + str(lbk))
        print("ACC = " + str(acc))
        print("poziom bledu = " + str(err))
        print("czulosc = " + str(czulosc))
        print("specyficznosc = " + str(specyficznosc))
        print("precyzja przewidywania pozytywnego = " + str(ppv))
        print("precyzja przewidywania negatywnego = " + str(npv))
        print("----------------------------------------------------")
        g_lpk += lpk
        g_lbk += lbk
        g_acc += acc
        g_err += err
        g_czulosc += czulosc
        g_specyficznosc += specyficznosc
        g_ppv += ppv
        g_npv += npv
    print("-----GLOBAL-----")
    print("liczba poprawnych klasyfikacji = " + str(g_lpk/NUMBER_OF_IMAGES))
    print("liczba blednych klasyfikacji = " + str(g_lbk/NUMBER_OF_IMAGES))
    print("ACC = " + str(g_acc/NUMBER_OF_IMAGES))
    print("poziom bledu = " + str(g_err/NUMBER_OF_IMAGES))
    print("czulosc = " + str(g_czulosc/NUMBER_OF_IMAGES))
    print("specyficznosc = " + str(g_specyficznosc/NUMBER_OF_IMAGES))
    print("precyzja przewidywania pozytywnego = " + str(g_ppv/NUMBER_OF_IMAGES))
    print("precyzja przewidywania negatywnego = " + str(g_npv/NUMBER_OF_IMAGES))