import os
import sys
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage

import click
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from .class_config import InferenceConfig, OneClassConfig, OneClassDataset
from .constant import (INITIAL_EPOCHS, INITIAL_LR, INPUT_SIZE, SECOND_EPOCHS,
                       SECOND_LR, IMAGE_SIZE, INPUT_SIZE)
from .generator import dataset_generator
from .util import check_dataset, get_ax, mask2image


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset_dir', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--out_dir', '-o', default='logs',
              help='Dataset folder path')
def prepare(dataset_dir, out_dir):
    out_dir = os.path.join(dataset_dir, out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    dataset_dir = os.path.abspath(dataset_dir)
    out_dir = os.path.abspath(out_dir)

    dataset_generator(dataset_dir, out_dir)
    check_dataset(out_dir)
    print("Generate dataset")


@cli.command()
@click.option('--dataset_path', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--out_dir', '-o', default='logs',
              help='Dataset folder path')
@click.option('--reset', '-r', is_flag=True,
              help='start training from coco weights')
def train(dataset_path, out_dir, reset):
    dataset_dir = os.path.abspath(dataset_path)
    out_dir = os.path.join(dataset_dir, out_dir)
    # Training dataset
    TRAIN_DATASET = os.path.join(out_dir, 'train')
    dataset_train = OneClassDataset()
    dataset_train.load_dataset(TRAIN_DATASET)
    dataset_train.prepare()

    # Validation dataset
    VALID_DATASET = os.path.join(out_dir, 'valid')
    dataset_val = OneClassDataset()
    dataset_val.load_dataset(VALID_DATASET)
    dataset_val.prepare()

    config = OneClassConfig()

    # Create model in training mode
    MODEL_DIR = os.path.join(out_dir, "model")

    if reset and os.path.exists(MODEL_DIR):
        print("Remove previous model dir")
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR)
    elif not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    else:
        print("Resume training from past weights")

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    try:
        weights_path = model.find_last()
    except FileNotFoundError:
        weights_path = None

    if weights_path:
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)
        print("Training restart")
        model.train(dataset_train, dataset_val,
                    learning_rate=SECOND_LR,
                    epochs=SECOND_EPOCHS,
                    layers='all')
    else:
        print("No previous model found: Use coco weights")
        COCO_MODEL_PATH = os.path.join('src', 'mask_rcnn_coco.h5')
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=INITIAL_LR,
                    epochs=INITIAL_EPOCHS,
                    layers='heads')


@cli.command()
@click.option('--image_path', '-i', required=True,
              help='image folder path',
              type=click.Path(exists=True))
@click.option('--model_dir', '-m', default='dataset/logs/model',
              help='trained weights model dir')
@click.option('--out_dir', '-o', default='dataset/logs/out',
              help='log dir after training')
def inference(image_path, model_dir, out_dir):
    image_path = os.path.abspath(image_path)
    model_dir = os.path.abspath(model_dir)
    out_dir = os.path.abspath(out_dir)

    if not os.path.exists(model_dir):
        print(f"Error: model dir {model_dir} not exists")
        sys.exit()
    elif not os.path.exists(image_path):
        print(f"Error: Image path {image_path} not exists")
        sys.exit()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                              config=config)
    weights_path = model.find_last()
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    image = skimage.io.imread(image_path)
    image = image[..., np.newaxis]
    results = model.detect([image], verbose=1)[0]

    image_pred = mask2image(results['masks'])
    image_pred = Image.fromarray(np.uint8(image_pred)).convert('L')
    image_pred.save(os.path.join(out_dir, os.path.basename(image_path)))


@cli.command()
@click.option('--dataset_path', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--out_dir', '-o', default='logs',
              help='name of output dir')
def validation(dataset_path, out_dir):
    """Valdiationデータについて推論実行
    """
    dataset_dir = os.path.abspath(dataset_path)
    out_dir = os.path.join(dataset_dir, out_dir)

    # Validation dataset
    VALID_DATASET = os.path.join(out_dir, 'valid')
    dataset_val = OneClassDataset()
    dataset_val.load_dataset(VALID_DATASET)
    dataset_val.prepare()

    config = InferenceConfig()
    MODEL_DIR = os.path.join(out_dir, "model")
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    weights_path = model.find_last()
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    result_dir = os.path.join(out_dir, "valid_result")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    for image_id in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config, image_id)
        info = dataset_val.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(
            info["source"], info["id"], image_id,
            dataset_val.image_reference(image_id)))

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_val.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        name = info['path'].stem
        plt.savefig(os.path.join(result_dir, f'{name}.jpg'))
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        plt.close()


if __name__ == '__main__':
    cli()
