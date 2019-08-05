import copy
import glob
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import cv2
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import log

from .constant import (COCO_MODEL_PATH, DATASET_NAME, IMAGES_PER_GPU,
                       OBJECT_NAME, ROOT_DIR)


def blob_detection(mask_path):
    mask = cv2.imread(mask_path, 0)
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    label = cv2.connectedComponentsWithStats(mask)
    data = copy.deepcopy(label[1])

    labels = []
    for label in np.unique(data):
        if label == 0:
            continue
        area = (data == label).sum()
        if area > 200:
            labels.append(label)

    mask = np.zeros((mask.shape)+(len(labels),), dtype=np.uint8)

    for n, label in enumerate(labels):
        mask[:, :, n] = np.uint8(data == label)

    cls_idxs = np.ones([mask.shape[-1]], dtype=np.int32)

    return mask, cls_idxs


class OneClassConfig(Config):

    #: config名
    NAME = DATASET_NAME

    #: batchあたりの画像数
    #: GPUのメモリが大きいなら増やしてもよい
    IMAGES_PER_GPU = IMAGES_PER_GPU

    # クラス数　= 背景 + 検出クラス数
    NUM_CLASSES = 1 + 1

    # エポックあたりステップ数
    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class OneClassDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        """ 1クラス検出のためのDatasetクラス
            self.image_infoへ元画像とマスク画像へのパスを登録する
            dataset_dirはtrainとmaskサブフォルダを持ちそこに画像が格納されている

        """
        #: データセット登録
        self.add_class(DATASET_NAME, 1, OBJECT_NAME)

        images = glob.glob(os.path.join(dataset_dir, "image", "*.jpg"))
        masks = glob.glob(os.path.join(dataset_dir, "mask", "*.jpg"))
        assert len(images) == len(masks)
        assert len(images) and len(masks), "jpg画像が存在しない"

        for image_path, mask_path in zip(images, masks):
            image_path = pathlib.Path(image_path)
            mask_path = pathlib.Path(mask_path)
            assert image_path.name == mask_path.name, 'データセット名不一致'

            image = Image.open(image_path)
            height = image.size[0]
            width = image.size[1]

            mask = Image.open(mask_path)
            assert image.size == mask.size, "サイズ不一致"

            self.add_image(
                DATASET_NAME,
                path=image_path,
                image_id=image_path.stem,
                mask_path=mask_path,
                width=width, height=height)

    def load_mask(self, image_id):
        """マスクデータとクラスidを生成する
        Returns:
          masks: Bool array [height, width, 物体数]
          class_ids: クラスidの1Darray
                     1クラス検出だからすべてid=1なのでnp.ones(物体数)
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != DATASET_NAME:
            return super(self.__class__, self).load_mask(image_id)

        mask_path = image_info['mask_path']
        mask, cls_idxs = blob_detection(str(mask_path))

        #: 領域重複の対策
        #: 重なっていたら上のレイヤーを優先する
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(mask.shape[-1]-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        return mask, cls_idxs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == DATASET_NAME:
            return info
        else:
            super(self.__class__, self).image_reference(image_id)


class InferenceConfig(OneClassConfig):

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.4


if __name__ == '__main__':
    print(COCO_MODEL_PATH)
    config = InferenceConfig()
    config.display()
