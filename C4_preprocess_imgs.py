import argparse
import json

from PIL import Image
import numpy as np

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

from tqdm import tqdm
import torch

from C4_helper import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
args, unk = parser.parse_known_args()
config = json.load(open(args.config))

verbose = True if config['verbose'] == 'True' else False

## sources bbox/grabCut:
# √ https://github.com/arunponnusamy/cvlib 
# √ https://github.com/lihini223/Object-Detection-model/blob/main/objectdetection.ipynb
# √ https://www.analyticsvidhya.com/blog/2022/03/image-segmentation-using-opencv/, oct 19, 19:59

# ? # faster r-cnn in pytorch: http://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

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

    # Normalise output: Checking if get_bbox was succesful occurs based on 
    # len of bbox list. Images with 0 cats (bbox list len<0), and images with
    # multiple cats (bbox list len>1) are excluded.
    use_img = False

    if len(bbox)>0:
        # Reset bbox, if one object was detected, but was not a cat.
        if len(bbox)==1 and label[0]=='cat':  #TBD check confidence scores
            use_img = True
        # If only one cat among multiple objects, drop non-cat objects.
        elif len(bbox)>1:
            if sum([1 for box_label in label if box_label=='cat']) == 1:
                cat_idx = label.index('cat')
                bbox = [bbox[cat_idx],]
                label = [label[cat_idx],]
                conf = [conf[cat_idx],]
                use_img = True

    # Create deepcopy to avoid editing original image in-place.
    if use_img:
        img_copy = np.copy(img)
        output_image = draw_bbox(img_copy, bbox, label, conf)

        return use_img, output_image, bbox
    else:
        return use_img, None, bbox

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


def get_cropped_ROIs(data: DataLoader, verbose: bool = False,
                     toy: bool = False) -> dict:
    """Read and resize images, run object detetion and judge suitablility
    of image for further use. If suitable, use bounding box to run grabCut
    algorithm and return dict of image ROIs and pruned images."""
    print('Preprocessing images...')

    size_to = (int(data.avg_size[0]/2), int(data.avg_size[1]/2))
    data = data.imgs[1200:] if toy else data.imgs

    # compressed_ROIs = list()
    img_dict = {'cropped_imgs': {'img_arrs': list(), 'filenames': list()},
                'detect_fail': list(),
                'multi_obj': list()}

    for imgfile in tqdm(data):
        # Read and resize image to half its size to improve runtime
        with Image.open(imgfile) as f:
            img = np.array(f.resize(size_to))

        # Run object detection, evaluate output for further processing.
        use_img, _, bbox = get_bbox(img)
        ## TBD work with labels and conf thresholds?
        if use_img:
            cut_image = grabcut_algorithm(img, bbox[0])
            img_dict['cropped_imgs']['img_arrs'].append(cut_image)
            img_dict['cropped_imgs']['filenames'].append(imgfile)
        else:
            if len(bbox) > 1:
                # Failure, more than 1 cat
                img_dict['multi_obj'].append(imgfile)
            else:
                # Failure, no cat
                img_dict['detect_fail'].append(imgfile)

    if verbose:
        print('Images excluded:', (len(img_dict['detect_fail'])
                                   + len(img_dict['multi_obj'])))
        print('Due to object detection failure:', len(img_dict['detect_fail'])) #131
        print('Due to multiple objects found:', len(img_dict['multi_obj']))

    return img_dict


if __name__=='__main__':
    data = DataLoader(config['toy_dir'])
    img_dict = get_cropped_ROIs( data, verbose=config['verbose'], toy=False)
 
    if config['ROI_nps_file']:
        np.save(config['ROI_nps_file'], img_dict['cropped_imgs']['img_arrs'])