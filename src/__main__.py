import os
import shutil
import random

import click
import matplotlib.pyplot as plt
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import visualize

from .class_config import InferenceConfig, OneClassConfig, OneClassDataset
from .constant import INITIAL_EPOCHS, INITIAL_LR, INPUT_SIZE
from .generator import dataset_generator
from .util import check_dataset, get_ax


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
def train(dataset_path, out_dir):
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
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    COCO_MODEL_PATH = os.path.join('src','mask_rcnn_coco.h5')
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=INITIAL_LR,
                epochs=INITIAL_EPOCHS,
                layers='heads')


@cli.command()
@click.option('--images_dir', '-i', required=True,
              help='image folder path',
              type=click.Path(exists=True))
@click.option('--logs', '-l', default='dataset/logs',
              help='log dir after training')
def predict(images_dir, logs):
    images_dir = os.path.abspath(images_dir)
    logs_dir = os.path.abspath(logs)
    print(images_dir)
    print(logs_dir)


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


if __name__ == '__main__':
    cli()
