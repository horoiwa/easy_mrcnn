import click
import os
import shutil
import mrcnn.model as modellib

from .generator import dataset_generator
from .util import check_dataset
from .class_config import OneClassConfig, OneClassDataset
from .constant import INITIAL_EPOCHS, INITIAL_LR


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
def training(dataset_path, out_dir):
    dataset_dir = os.path.abspath(dataset_path)
    out_dir = os.path.join(dataset_dir, out_dir)
    # Training dataset
    TRAIN_DATASET = "dataset/logs/train"
    dataset_train = OneClassDataset()
    dataset_train.load_dataset(TRAIN_DATASET)
    dataset_train.prepare()

    # Validation dataset
    VALID_DATASET = "dataset/logs/train"
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

    COCO_MODEL_PATH = 'src/mask_rcnn_coco.h5'
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=INITIAL_LR,
                epochs=INITIAL_EPOCHS,
                layers='heads')

if __name__ == '__main__':
    cli()
