import argparse
import json

import os
import pandas as pd

from PIL import Image
import numpy as np

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

from tqdm import tqdm
import torch

from sklearn_col_compressor import compress_colours


class DataLoader:
    def __init__(self, data_dir: str, split_data: bool = False):
        self.imgs, self.points = self._read_files(data_dir)
        if split_data:
            self.train, self.dev, self.test = self._split()
            self.avg_size = self._get_avg_size(self.train['imgs'])
        else:
            self.avg_size = self._get_avg_size(self.imgs)

    def _read_files(self, data_dir):
        catfiles, pointfiles = list(), list()
        for root, dirs, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith('jpg'):
                    catfiles.append(os.path.join(root, fname))
                elif fname.endswith('cat'):
                    pointfiles.append(os.path.join(root, fname))

        return catfiles, pointfiles

    def _split(self):
        data_df = pd.DataFrame({'imgs': self.imgs, 'points': self.points})

        # Set cutoff points.
        train_len = int(len(data_df)*0.8)
        dev_len = int(train_len + len(data_df)*0.1)

        # Split data.
        train_data = data_df[:train_len]
        dev_data = data_df[train_len:dev_len]
        test_data = data_df[dev_len:]

        return train_data, dev_data, test_data

    def __len__(self):
        return len(self.imgs)

    def _get_avg_size(self, data: list) -> tuple[int, int]:
        """Find average size of training files to resize input images to."""
        # Collect file widths and heights.
        sizes_w, sizes_h = list(), list()
        for file in data:
            with Image.open(file) as img:
                size = img.size
                sizes_w.append(size[0])
                sizes_h.append(size[1])
        # Get average file size in training dataset.
        avg_size = (round(sum(sizes_w) / len(data)),
                    round(sum(sizes_h) / len(data)))

        return avg_size

## sources bbox/grabCut:
# √ https://github.com/arunponnusamy/cvlib 
# √ https://github.com/lihini223/Object-Detection-model/blob/main/objectdetection.ipynb
# √ https://www.analyticsvidhya.com/blog/2022/03/image-segmentation-using-opencv/, oct 19, 19:59

def get_bbox(imgfile: str) -> tuple[np.ndarray, list]:

    if type(imgfile)==str:
        with Image.open(imgfile) as f:
            img = np.array(f)
    else:
        img = np.copy(imgfile)
        #np.float32(imgfile)
    #img = cv2.imread(imgfile)
    #img = cv2.resize(img ,(500,500))

    cuda_status = True if torch.cuda.is_available() else False

    # confidence? nms tresh? ? gpu?
    # models:  yolov3, yolov3-tiny, yolov4, yolov4-tiny
    bbox, label, conf = cv.detect_common_objects(img, model='yolov3',
                                                 enable_gpu=cuda_status)
    img_copy = np.copy(img)
    if len(bbox)>1:
        if sum([1 for box_label in label if box_label=='cat']) == 1:
            cat_idx = label.index('cat')
            bbox = [bbox[cat_idx],]
            label = [label[cat_idx],]
            conf = [conf[cat_idx],]
        #else:
         #   print(label, conf)

    output_image = draw_bbox(img_copy, bbox, label, conf)

    return output_image, bbox

def grabcut_algorithm(img: str, bounding_box: list, iterations: int = 2) \
                        -> np.ndarray:

    # more advanced segmentation based on colour in cv2?
    #https://www.kaggle.com/code/amulyamanne/image-segmentation-color-clustering/notebook

    if type(img)==str:
        with Image.open(img) as f:
            img_arr = np.array(f, dtype=np.uint8)

    elif img.dtype == np.float32 or img.dtype == np.float64:
        img_arr = np.array(img*255, dtype=np.uint8)
    else:
        img_arr = img

    segment = np.zeros(img_arr.shape[:2], np.uint8)

    x,y,width,height = bounding_box
    segment[y:y+height, x:x+width] = 1

    background_mdl = np.zeros((1,65), np.float64)
    foreground_mdl = np.zeros((1,65), np.float64)

    # prioritise low iter for speed
    cv2.grabCut(img_arr, segment, bounding_box, background_mdl,
                foreground_mdl, iterations, cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((segment==2)|(segment==0),0,1).astype('uint8')

    cut_img = img_arr*new_mask[:,:,np.newaxis]

    return cut_img

def get_compressed_ROIs(data: DataLoader, verbose: bool = False,
                        toy: bool = False) -> list:

    size_to = (int(data.avg_size[0]/2), int(data.avg_size[1]/2))
    data = data.imgs[:300] if toy else data.imgs

    compressed_ROIs = list()
    box_fail, multi_obj, col_fail = list(), list(), list()

    for imgfile in tqdm(data):
        with Image.open(imgfile) as f:
            img = np.array(f.resize(size_to))

        _, bbox = get_bbox(img)
        ## TBD work with labels and conf thresholds?
        if len(bbox)==1:
            cut_image = grabcut_algorithm(img, bbox[0])
            compressed_ROI = compress_colours(cut_image, n_colours=64)

            if type(compressed_ROI) == np.ndarray:
                compressed_ROIs.append(compressed_ROI)
            else:
                col_fail.append(imgfile)

        elif len(bbox) > 1:
            #[cut_images.append(grabcut_algorithm(imgfile, bbox_i)) for bbox_i in bbox ]
            multi_obj.append(imgfile)
        else:
            box_fail.append(imgfile)
    if verbose:
        print('Images excluded:', len(box_fail)+len(multi_obj)+len(col_fail))
        print('Due to object detection failure:', len(box_fail)) #131
        print('Due to multiple objects found:', len(multi_obj))
        print('Due to insufficient colour clusters:', len(col_fail))

        return box_fail, multi_obj, col_fail, compressed_ROIs
    else:
        return compressed_ROIs

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration file',
                        default='config.json')
    args = parser.parse_args()
    config = json.load(open(args.config))

    verbose = config['verbose']
    data = DataLoader(config['toy_dir'])

    if verbose=='True':
        bbox_err, multi_err, col_err, img_ROIs = get_compressed_ROIs(
            data, verbose=True, toy=False)
    else:
        img_ROIs = get_compressed_ROIs(data, verbose=False,
                                 toy=False)

    if config['ROI_nps_file']:
        np.save(config['ROI_nps_file'], img_ROIs)