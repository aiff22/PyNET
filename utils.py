# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from functools import reduce
import tensorflow as tf
import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_ITERS = [100000, 35000, 20000, 20000, 5000, 5000]


def process_command_args(arguments):

    # Specifying the default parameters

    level = 0
    batch_size = 50

    train_size = 5000
    learning_rate = 5e-5

    eval_step = 1000
    restore_iter = None
    num_train_iters = None

    dataset_dir = 'raw_images/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'

    for args in arguments:

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])

    if restore_iter is None and level < 5:
        restore_iter = get_last_iter(level + 1)
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level + 1) + ".")
            print("Aborting the training.")
            sys.exit()

    if num_train_iters is None:
        num_train_iters = NUM_DEFAULT_TRAIN_ITERS[level]

    print("The following parameters will be applied for CNN training:")

    print("Training level: " + str(level))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(learning_rate))
    print("Training iterations: " + str(num_train_iters))
    print("Evaluation step: " + str(eval_step))
    print("Restore Iteration: " + str(restore_iter))
    print("Path to the dataset: " + dataset_dir)
    print("Path to VGG-19 network: " + vgg_dir)

    return level, batch_size, train_size, learning_rate, restore_iter, num_train_iters,\
           dataset_dir, vgg_dir, eval_step


def process_test_model_args(arguments):

    level = 0
    restore_iter = None

    dataset_dir = 'raw_images/'
    use_gpu = "true"

    orig_model = "false"

    for args in arguments:

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("orig"):
            orig_model = args.split("=")[1]

    if restore_iter is None and orig_model == "false":
        restore_iter = get_last_iter(level)
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level) + ".")
            sys.exit()

    return level, restore_iter, dataset_dir, use_gpu, orig_model


def get_last_iter(level):

    saved_models = [int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir("models/")
                    if model_file.startswith("pynet_level_" + str(level))]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return -1


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

