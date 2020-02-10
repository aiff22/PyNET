# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
from scipy import misc
import numpy as np
import sys

from load_dataset import load_training_batch, load_test_data
from model import PyNET
import utils
import vgg

# Processing command arguments

LEVEL, batch_size, train_size, learning_rate, restore_iter, num_train_iters, dataset_dir, vgg_dir, eval_step = \
    utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches

PATCH_WIDTH, PATCH_HEIGHT = 224, 224

DSLR_SCALE = float(1) / (2 ** (LEVEL - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)

# Defining the model architecture

with tf.Graph().as_default(), tf.Session() as sess:
    
    # Placeholders for training data

    phone_ = tf.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_ = tf.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image

    output_l0, output_l1, output_l2, output_l3, output_l4, output_l5 = \
        PyNET(phone_, instance_norm=True, instance_norm_level_1=False)

    if LEVEL == 5:
        enhanced = output_l5
    if LEVEL == 4:
        enhanced = output_l4
    if LEVEL == 3:
        enhanced = output_l3
    if LEVEL == 2:
        enhanced = output_l2
    if LEVEL == 1:
        enhanced = output_l1
    if LEVEL == 0:
        enhanced = output_l0

    # Losses

    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])

    # MSE loss
    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2))/(TARGET_SIZE * batch_size)

    # PSNR loss
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # SSIM loss
    loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))

    # MS-SSIM loss
    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    # Content loss
    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # Final loss function

    if LEVEL == 5 or LEVEL == 4:
        loss_generator = loss_mse * 100
    if LEVEL == 3 or LEVEL == 2:
        loss_generator = loss_mse * 100 + loss_content
    if LEVEL == 1:
        loss_generator = loss_mse * 50 + loss_content
    if LEVEL == 0:
        loss_generator = loss_mse * 20 + loss_content + (1 - loss_ssim) * 20

    # Optimize network parameters

    generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    # Initialize and restore the variables

    print("Initializing variables")
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

    if LEVEL < 5:
        print("Restoring Variables")
        saver.restore(sess, "models/pynet_level_" + str(LEVEL + 1) + "_iteration_" + str(restore_iter) + ".ckpt")

    # Loading training and test data

    print("Loading test data...")
    test_data, test_answ = load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Test data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_training_batch(dataset_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Training data was loaded\n")

    TEST_SIZE = test_data.shape[0]
    num_test_batches = int(test_data.shape[0] / batch_size)

    visual_crops_ids = np.random.randint(0, TEST_SIZE, batch_size)
    visual_test_crops = test_data[visual_crops_ids, :]
    visual_target_crops = test_answ[visual_crops_ids, :]

    print("Training network")

    logs = open("models/logs.txt", "w+")
    logs.close()

    training_loss = 0.0

    for i in range(num_train_iters + 1):

        # Train PyNET model

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        # Random flips and rotations

        for k in range(batch_size):

            random_rotate = np.random.randint(1, 100) % 4
            phone_images[k] = np.rot90(phone_images[k], random_rotate)
            dslr_images[k] = np.rot90(dslr_images[k], random_rotate)
            random_flip = np.random.randint(1, 100) % 2

            if random_flip == 1:
                phone_images[k] = np.flipud(phone_images[k])
                dslr_images[k] = np.flipud(dslr_images[k])

        # Training step

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict={phone_: phone_images, dslr_: dslr_images})
        training_loss += loss_temp / eval_step

        if i % eval_step == 0:

            # Evaluate PyNET model

            test_losses = np.zeros((1, 5 if LEVEL < 2 else 4))

            for j in range(num_test_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                if LEVEL < 2:
                    losses = sess.run([loss_generator, loss_content, loss_mse, loss_psnr, loss_ms_ssim], \
                                    feed_dict={phone_: phone_images, dslr_: dslr_images})
                else:
                    losses = sess.run([loss_generator, loss_content, loss_mse, loss_psnr], \
                                      feed_dict={phone_: phone_images, dslr_: dslr_images})

                test_losses += np.asarray(losses) / num_test_batches

            if LEVEL < 2:
                logs_gen = "step %d | training: %.4g, test: %.4g | content: %.4g, mse: %.4g, psnr: %.4g, " \
                           "ms-ssim: %.4g\n" % (i, training_loss, test_losses[0][0], test_losses[0][1],
                                                test_losses[0][2], test_losses[0][3], test_losses[0][4])
            else:
                logs_gen = "step %d | training: %.4g, test: %.4g | content: %.4g, mse: %.4g, psnr: %.4g\n" % \
                       (i, training_loss, test_losses[0][0], test_losses[0][1], test_losses[0][2], test_losses[0][3])
            print(logs_gen)

            # Save the results to log file

            logs = open("models/logs.txt", "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # Save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: visual_test_crops, dslr_: dslr_images})

            idx = 0
            for crop in enhanced_crops:
                if idx < 4:
                    before_after = np.hstack((crop,
                                    np.reshape(visual_target_crops[idx], [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])))
                    misc.imsave("results/pynet_img_" + str(idx) + "_level_" + str(LEVEL) + "_iter_" + str(i) + ".jpg",
                                    before_after)
                idx += 1

            training_loss = 0.0

            # Saving the model that corresponds to the current iteration
            saver.save(sess, "models/pynet_level_" + str(LEVEL) + "_iteration_" + str(i) + ".ckpt", write_meta_graph=False)

        # Loading new training data
        if i % 1000 == 0:

            del train_data
            del train_answ
            train_data, train_answ = load_training_batch(dataset_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

