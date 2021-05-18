import os.path

import numpy as np
import glob
from PIL import Image


def get_images(path):
    image_paths = glob.glob(path + '/*.jpg')

    for image_file in image_paths:
        image = Image.open(image_file).resize((220, 220))
        image = np.asarray(image.convert('RGB')) / 255.0

        yield image


def get_image_data(path):
    images_data = []

    images = get_images(path)
    for img in images:
        images_data.append(img)

    return images_data


def get_txt_index(path, index):
    txt_data = []
    txt_path = glob.glob(path + '/*.txt')

    for txt_file in txt_path:
        with open(txt_file, 'r', encoding='utf-8') as f:
            tmp = f.readline().split(' ')
            tmp = int(tmp[index])

            if tmp == 2 and index == 0:
                tmp = 0
            elif index == 1 and index == 3:
                img = Image.open(os.path.splitext(txt_file)[0]+'.jpg')
                w, _ = img.size
                tmp = tmp / w
            elif index == 2 and index == 4:
                img = Image.open(os.path.splitext(txt_file)[0] + '.jpg')
                _, h = img.size
                tmp = tmp / h

            txt_data.append(tmp)

    return txt_data


def get_all_data(path):
    data = get_image_data(path)

    classes = get_txt_index(path, 0)
    xmin = get_txt_index(path, 1)
    ymin = get_txt_index(path, 2)
    xmax = get_txt_index(path, 3)
    ymax = get_txt_index(path, 4)

    data = np.array(data, dtype='float32')
    classes = np.array(classes, dtype='float32')
    xmin = np.array(xmin, dtype='float32')
    ymin = np.array(ymin, dtype='float32')
    xmax = np.array(xmax, dtype='float32')
    ymax = np.array(ymax, dtype='float32')

    targets = {'class': classes,
               'out': [xmin, ymin, xmax, ymax]}

    return data, targets
