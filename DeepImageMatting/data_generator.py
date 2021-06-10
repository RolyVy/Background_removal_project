import math
import os
import random
import gc
from random import shuffle

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import num_train_samples, num_valid_samples
from config import batch_size
from config import dataset_path, a_path, image_path, image_path_test, a_path_test
from config import img_cols, img_rows
from config import unknown_code
from utils import safe_crop

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        filename = dataset_path + ('{}_names.txt'.format(usage))

        with open(filename, 'r') as f:
            self.names = f.read().splitlines()#[:num_train_samples]
        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = (idx * batch_size)

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 4), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 2), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            name= name.split('.')[0]
            im_name = name + ".jpg"
            alpha_name = name + ".png"
            image = cv.imread(os.path.join(image_path,im_name))
            alpha = cv.imread(os.path.join(a_path,alpha_name),cv.IMREAD_UNCHANGED)


            #frame = cv2.resize(frame,(224,224),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            image = cv.resize(src=image, dsize=(img_cols, img_rows), interpolation=cv.INTER_CUBIC)
            alpha = cv.resize(src=alpha, dsize=(img_cols, img_rows), interpolation=cv.INTER_CUBIC)

            trimap = generate_trimap(alpha)

            # Flip array left to right randomly (prob=1:1)
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
                alpha = np.fliplr(alpha)

            batch_x[i_batch, :, :, 0:3] = image / 255.
            batch_x[i_batch, :, :, 3] = trimap / 255.

            mask = np.equal(trimap, 128).astype(np.float32)
            batch_y[i_batch, :, :, 0] = alpha / 255.
            batch_y[i_batch, :, :, 1] = mask
            del image,alpha,trimap

            i += 1
            test_gen()
        return batch_x, batch_y



class DataGenSequence_test(Sequence):
    def __init__(self, usage):
        self.usage = usage
        filename = dataset_path + ('{}_names.txt'.format(usage))
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()#[:num_train_samples]
        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = (idx * batch_size)

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 4), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 2), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            name= name.split('.')[0]
            im_name = name + ".jpg"
            alpha_name = name + ".png"
            image = cv.imread(os.path.join(image_path_test,im_name))
            alpha = cv.imread(os.path.join(a_path_test,alpha_name),cv.IMREAD_UNCHANGED)


            #frame = cv2.resize(frame,(224,224),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            image = cv.resize(src=image, dsize=(img_cols, img_rows), interpolation=cv.INTER_CUBIC)
            alpha = cv.resize(src=alpha, dsize=(img_cols, img_rows), interpolation=cv.INTER_CUBIC)

            trimap = generate_trimap(alpha)

            # Flip array left to right randomly (prob=1:1)
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
                alpha = np.fliplr(alpha)

            batch_x[i_batch, :, :, 0:3] = image / 255.
            batch_x[i_batch, :, :, 3] = trimap / 255.

            mask = np.equal(trimap, 128).astype(np.float32)
            batch_y[i_batch, :, :, 0] = alpha / 255.
            batch_y[i_batch, :, :, 1] = mask
            del image,alpha,trimap

            i += 1
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)
        gc.collect()


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')

#create test_gen
def test_gen():
    return DataGenSequence_test('test')  

#if __name__ == '__main__':
#     filename = 'data/input/2_1.png'
#     bgr_img = cv.imread(filename)
#     bg_h, bg_w = bgr_img.shape[:2]
#     print(bg_w, bg_h)