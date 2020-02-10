# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np

import model
import utils

from load_dataset import load_test_data

PATCH_WIDTH, PATCH_HEIGHT = 224, 224
TARGET_SIZE = (PATCH_WIDTH * 2) * (PATCH_HEIGHT * 2) * 3
config = None

# Path to the dataset:
dataset_dir = 'raw_images/'

# Disable gpu (if needed):
# config = tf.ConfigProto(device_count={'GPU': 0})

with tf.Session(config=config) as sess:

    # Create placeholders for input and target images

    phone_ = tf.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_HEIGHT, 4])
    dslr_ = tf.placeholder(tf.float32, [1, int(PATCH_HEIGHT * 2), int(PATCH_WIDTH * 2), 3])

    # Process raw images with your model:
    # enhanced = your_model(phone_), e.g.:
    enhanced, _, _, _, _, _ = model.PyNET(phone_, instance_norm=True, instance_norm_level_1=False)

    # Compute losses

    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])

    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2)) / TARGET_SIZE
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    saver = tf.train.Saver()

    # Restore your own model from a checkpoint
    # saver.restore(sess, "path_to_your_checkpoint"), e.g.:
    saver.restore(sess, "models/original/pynet_level_0.ckpt")

    print("Loading test data...")
    test_data, test_answ = load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, 2.0)
    print("Test data was loaded\n")

    loss_ssim_ = 0.0
    loss_psnr_ = 0.0

    test_size = test_data.shape[0]
    for j in range(test_size):

        if j % 100 == 0:
            print(j)

        phone_images = np.reshape(test_data[j], [1, PATCH_HEIGHT, PATCH_WIDTH, 4])
        dslr_images = np.reshape(test_answ[j], [1, int(PATCH_HEIGHT * 2), int(PATCH_WIDTH * 2), 3])

        losses = sess.run([loss_psnr, loss_ms_ssim], feed_dict={phone_: phone_images, dslr_: dslr_images})

        loss_psnr_ += losses[0]
        loss_ssim_ += losses[1]

    loss_psnr_ = float(loss_psnr_) / test_size
    loss_ssim_ = float(loss_ssim_) / test_size

    output_logs = "PSNR: %.4g, MS-SSIM: %.4g\n" % (loss_psnr_, loss_ssim_)
    print(output_logs)
