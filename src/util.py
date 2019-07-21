import os
import random
import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mrcnn.model import log
from mrcnn import utils
from mrcnn import visualize

from .class_config import OneClassDataset


def check_dataset(out_dir):
    train_dir = os.path.join(out_dir, 'train')

    dataset_train = OneClassDataset()
    dataset_train.load_dataset(train_dir)
    dataset_train.prepare()
    out_path = os.path.join(out_dir, 'train_dataset.png')
    show_instance(dataset_train, out_path)

    valid_dir = os.path.join(out_dir, 'valid')
    dataset_valid = OneClassDataset()
    dataset_valid.load_dataset(valid_dir)
    dataset_valid.prepare()
    out_path = os.path.join(out_dir, 'valid_dataset.png')
    show_instance(dataset_train, out_path)


def show_instance(dataset, out_path):
    # Load random image and mask.
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask,
                                class_ids, dataset.class_names)
    plt.savefig(out_path)
