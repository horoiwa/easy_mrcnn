import os
from mrcnn import utils

DATASET_NAME = 'cell_dataset'
OBJECT_NAME = 'cell'


IMAGES_PER_GPU = 1

INITIAL_EPOCHS = 1
INITIAL_LR = 0.001
SECOND_EPOCHS = 2
SECOND_LR = 0.001


ROOT_DIR = os.path.abspath('.')
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "src", "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_COLORMODE = 'L'
MASK_COLORMODE = 'L'

N_TRAIN = 3
IMAGE_SIZE = (768, 768)
INPUT_SIZE = (512, 512)

#: 基本的にはこの設定値なら影響がない
PCA_COLOR = True
PCA_COLOR_RANGE = (-0.2, 0.2)

DATA_GEN_ARGS = dict(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    vertical_flip=True,
    horizontal_flip=True,
    cval=0,
    fill_mode='constant')


if __name__ == '__main__':
    print(ROOT_DIR)
    print(os.path.exists(COCO_MODEL_PATH))
