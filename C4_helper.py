import os
import pandas as pd

from PIL import Image
import numpy as np

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

from tqdm import tqdm
import torch

from sklearn.cluster import KMeans


# C4 helper funcs
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
            size = Image.open(file).size
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

def get_bbox(imgfile: str, size_to: tuple[int, int] = (250,250)) -> tuple[np.ndarray, list]:
    """_summary_

    Args:
        imgfile (str): _description_
        size_to (tuple[int, int], optional): _description_. Defaults to (250,250).

    Returns:
        tuple[np.ndarray, list]: _description_
    """
    if type(imgfile)==str:
        img = np.copy(np.array(Image.open(imgfile)))
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
    output_image = draw_bbox(img_copy, bbox, label, conf)

    return output_image, bbox

def grabcut_algorithm(img: str, bounding_box: list,
                      size_to: tuple[int, int] = (250,250)) -> np.ndarray:
    """_summary_

    Args:
        imgfile (str): _description_
        bounding_box (list): _description_
        size_to (tuple[int, int], optional): _description_. Defaults to (250,250).

    Returns:
        np.ndarray: _description_
    """
    # more advanced segmentation based on colour in cv2?
    #https://www.kaggle.com/code/amulyamanne/image-segmentation-color-clustering/notebook

    if type(img)==str:
        img_arr = np.array(Image.open(img), dtype=np.uint8)

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
                foreground_mdl, 2, cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((segment==2)|(segment==0),0,1).astype('uint8')

    cut_img = img_arr*new_mask[:,:,np.newaxis]

    return cut_img

def get_ROIs(data: DataLoader) -> list:
    """_summary_

    Args:
        data (DataLoader): _description_

    Returns:
        list: _description_
    """
    box_fail, multi_obj = 0, 0
    cut_images = list()

    for imgfile in tqdm(data.imgs):
        _, bbox = get_bbox(imgfile, data.avg_size)
        ## TBD work with labels and conf thresholds
        if len(bbox)==1:
            #plt.imshow(img)
            cut_images.append(grabcut_algorithm(imgfile, bbox[0],
                                                data.avg_size))
        elif len(bbox) > 1:
            #[cut_images.append(grabcut_algorithm(imgfile, bbox_i)) for bbox_i in bbox ]
            multi_obj+=1
        else:
            box_fail+=1

    print('Images excluded:', multi_obj+box_fail)
    print('Due to object detection failure:', box_fail) #131
    print('Due to multiple objects found:', multi_obj)

    return cut_images

def get_cluster(ROI_list, n=2):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb
    ROI_matrix = np.array([img_arr.flatten() for img_arr in ROI_list],
                          dtype=np.float32)
    # normalise values
    ROI_matrix /= 255
    #flat_ROIs = KMeans.fit_transform(flat_ROIs) ??

    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(ROI_matrix) #2mins for n=5, 50sec for n=2

    clusters_dict = {cluster_id : list() for cluster_id in set(Y)}
    for i, cluster_id in enumerate(Y):
        clusters_dict[cluster_id].append(ROI_list[i])

    return Y, clusters_dict