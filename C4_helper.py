import os
import pandas as pd

from PIL import Image
import numpy as np

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

from tqdm import tqdm

# C4 helper funcs
class DataLoader:
    def __init__(self, data_dir: str):
        self.train, self.dev, self.test = self._read_files(data_dir)

    def _read_files(self, data_dir):
        catfiles, pointfiles = list(), list()
        for root, dirs, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith('jpg'):
                    catfiles.append(os.path.join(root, fname))
                elif fname.endswith('cat'):
                    pointfiles.append(os.path.join(root, fname))

        data_df = pd.DataFrame({'imgs': catfiles, 'points': pointfiles})

        # Set cutoff points.
        train_len = int(len(data_df)*0.8)
        dev_len = int(train_len + len(data_df)*0.1)

        # Split data.
        train_data = data_df[:train_len]
        dev_data = data_df[train_len:dev_len]
        test_data = data_df[dev_len:]

        return train_data, dev_data, test_data

    def __len__(self):
        return len(self.train) + len(self.dev) + len(self.test)

## sources bbox/grabCut:
# √ https://github.com/arunponnusamy/cvlib 
# √ https://github.com/lihini223/Object-Detection-model/blob/main/objectdetection.ipynb
# √ https://www.analyticsvidhya.com/blog/2022/03/image-segmentation-using-opencv/, oct 19, 19:59

def get_bbox(imgfile: str) -> tuple[np.ndarray, list]:
    """_summary_

    Args:
        imgfile (str): _description_

    Returns:
        tuple[np.NDArray, list]: _description_
    """
    img = np.array(Image.open(imgfile).resize((500,500)))
    #img = cv2.imread(imgfile)
    #img = cv2.resize(img ,(500,500))

    # confidence? nms tresh? ? gpu?
    # models:  yolov3, yolov3-tiny, yolov4, yolov4-tiny
    bbox, label, conf = cv.detect_common_objects(img, model='yolov3') # enable_gpu=True
    output_image = draw_bbox(img, bbox, label, conf)

    return output_image, bbox[0]

def grabcut_algorithm(imgfile: str, bounding_box: list) -> np.ndarray:
    """_summary_

    Args:
        imgfile (str): _description_
        bounding_box (list): _description_

    Returns:
        np.ndarray: _description_
    """
    img = np.array(Image.open(imgfile).resize((500,500)))

    segment = np.zeros(img.shape[:2],np.uint8)

    x,y,width,height = bounding_box
    segment[y:y+height, x:x+width] = 1

    background_mdl = np.zeros((1,65), np.float64)
    foreground_mdl = np.zeros((1,65), np.float64)

    # prioritise low iter for speed
    cv2.grabCut(img, segment, bounding_box, background_mdl,
                foreground_mdl, 2, cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((segment==2)|(segment==0),0,1).astype('uint8')

    cut_img = img*new_mask[:,:,np.newaxis]

    return cut_img

def get_ROIs(data: pd.DataFrame) -> list:
    """_summary_

    Args:
        data (pd.DataFrame): _description_

    Returns:
        list: _description_
    """
    box_fail = 0
    cut_images = list()

    for imgfile in tqdm(data.train['imgs']):
        _, bbox = get_bbox(imgfile)
        if bbox:
            #plt.imshow(img)
            cut_images.append(grabcut_algorithm(imgfile, bbox))
        else:
            box_fail+=1
    print('failed to grab bbox on {} items.', format(box_fail)) #131

    return cut_images
