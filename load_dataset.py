# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from __future__ import print_function
from scipy import misc
from PIL import Image
import imageio
import os
import numpy as np


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


def load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):

    test_directory_dslr = dataset_dir + 'test/canon/'
    test_directory_phone = dataset_dir + 'test/huawei_raw/'

    # NUM_TEST_IMAGES = 1204
    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, 4))
    test_answ = np.zeros((NUM_TEST_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    for i in range(0, NUM_TEST_IMAGES):

        I = np.asarray(imageio.imread((test_directory_phone + str(i) + '.png')))
        I = extract_bayer_channels(I)
        test_data[i, :] = I
        
        I = np.asarray(Image.open(test_directory_dslr + str(i) + '.jpg'))
        I = misc.imresize(I, DSLR_SCALE / 2, interp='bicubic')
        I = np.float16(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        test_answ[i, :] = I

    return test_data, test_answ


def load_training_batch(dataset_dir, TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):

    train_directory_dslr = dataset_dir + 'train/canon/'
    train_directory_phone = dataset_dir + 'train/huawei_raw/'

    # NUM_TRAINING_IMAGES = 46839
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, 4))
    train_answ = np.zeros((TRAIN_SIZE, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    i = 0
    for img in TRAIN_IMAGES:

        I = np.asarray(imageio.imread((train_directory_phone + str(img) + '.png')))
        I = extract_bayer_channels(I)
        train_data[i, :] = I

        I = np.asarray(Image.open(train_directory_dslr + str(img) + '.jpg'))
        I = misc.imresize(I, DSLR_SCALE / 2, interp='bicubic')
        I = np.float16(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        train_answ[i, :] = I

        i += 1

    return train_data, train_answ

