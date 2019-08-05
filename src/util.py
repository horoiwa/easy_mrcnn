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
    print("Dataset train:", len(dataset_train.image_ids))
    out_path = os.path.join(out_dir, 'train_dataset.png')
    show_instance(dataset_train, out_path)

    valid_dir = os.path.join(out_dir, 'valid')
    dataset_valid = OneClassDataset()
    dataset_valid.load_dataset(valid_dir)
    dataset_valid.prepare()
    print("Dataset valid:", len(dataset_valid.image_ids))
    out_path = os.path.join(out_dir, 'valid_dataset.png')
    show_instance(dataset_valid, out_path)


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


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def mask2image(mask):
    """ maskを受け取り二値化画像を返す
        maskの重複が起こらないように処理している
    """
    image = np.zeros(mask.shape[:2])

    #: 領域重複の対策: 重なっていたら上のレイヤーを優先する
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(mask.shape[-1]-2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

    #: 重なるのが本気で嫌ならここで各masklayerにモルフォロジー演算で縮小かければよい
    for i in range(0, mask.shape[-1]):
        image = image + mask[:, :, i]

    return image*255

