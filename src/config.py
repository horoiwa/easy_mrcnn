import os
import sys

from constant import ROOT_DIR, COCO_MODEL_PATH
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN", "samples", "coco"))
import coco


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
    print(COCO_MODEL_PATH)
    config = InferenceConfig()
    config.display()
