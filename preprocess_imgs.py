import argparse
import json
import os

import numpy as np
from PIL import Image

import cv2
import cvlib as cv
from cvlib import detect_common_objects
from cvlib.object_detection import draw_bbox
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-srcdir', '--source_directory',
                    help='directory to run preprocessing on',
                    default=False)
parser.add_argument('-lim', '--limit',
                    help='limit number of images processed from dir',
                    default=False)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

LIMIT = int(args.limit) if args.limit != "False" else False

## sources bbox/grabCut:
# √ https://github.com/arunponnusamy/cvlib 
# √ https://github.com/lihini223/Object-Detection-model/blob/main/objectdetection.ipynb
# √ https://www.analyticsvidhya.com/blog/2022/03/image-segmentation-using-opencv/, oct 19, 19:59

# ? # faster r-cnn in pytorch: http://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

def get_avg_size(files: list) -> tuple[int, int]:
    """Find average size of training files to resize input images to."""
    # Collect file widths and heights.
    sizes_w, sizes_h = list(), list()
    for file in files:
        with Image.open(file) as img:
            size = img.size
            sizes_w.append(size[0])
            sizes_h.append(size[1])
    # Get average file size in training dataset.
    avg_size = (round(sum(sizes_w) / len(files)),
                round(sum(sizes_h) / len(files)))

    return avg_size


def get_bbox(imgfile: str|np.ndarray) -> tuple[bool, np.ndarray, list]:
    """Take an image (either as a filename or already as a numpy error)
    and run object detection with YoloV3. Retun a value to indicate whether
    to continue using the image (if exactly one cat is found), or whether
    to exclude it (zero or more than two cats found). Also return the
    image with a drawn bounding box, as well as the box' coordinates.

    Args:
        imgfile (str | np.ndarray): Filename or array of image.

    Returns:
        tuple[bool, np.ndarray, list]: Boolean to indicate suitability of
            image for further code, array of image with bounding box,
            bounding box coordinates.
    """

    if type(imgfile)==str:
        with Image.open(imgfile) as f:
            img = np.array(f)
    else:
        img = np.copy(imgfile)

    cuda_status = True if torch.cuda.is_available() else False

    # confidence? nms tresh? ? gpu?
    # models:  yolov3, yolov3-tiny, yolov4, yolov4-tiny
    bbox, label, conf = detect_common_objects(img, model='yolov4',
                                                 enable_gpu=cuda_status)

    # Check/edit output: Checking if get_bbox was succesful occurs based on 
    # len of bbox list. Images with 0 cats (bbox list len<0), and images with
    # multiple cats (bbox list len>1) are excluded. If exactly one object
    # of label cat was detected, set use_img to True.
    use_img = False

    if len(bbox)>0:
        # Reset bbox, if one object was detected, but was not a cat.
        if len(bbox)==1 and label[0]=='cat':  #TBD check confidence scores?
            use_img = True
        # If only one cat among multiple objects, drop non-cat objects.
        elif len(bbox)>1:
            if sum([1 for box_label in label if box_label=='cat']) == 1:
                cat_idx = label.index('cat')
                bbox = [bbox[cat_idx],]
                label = [label[cat_idx],]
                conf = [conf[cat_idx],]
                use_img = True

    # Create deepcopy to avoid editing original image object.
    if use_img:
        img_copy = np.copy(img)
        output_image = draw_bbox(img_copy, bbox, label, conf)
        return use_img, output_image, bbox

    else:
        return use_img, None, bbox


def grabcut_algorithm(img: str|np.ndarray, bounding_box: list,
                      iterations: int = 2) -> np.ndarray:
    """Take an image (as filename or array) and bounding box coordinates
    for one object within, and run the grabCut algorithm (with n 
    iterations) to separate the object (foregound) from the background.
    Return the thus isolated region of interest (ROI).

    Args:
        img (str|np.ndarray):  Filename or array of image.
        bounding_box (list): Coordinates of bounding box for image.
        iterations (int, optional): Number of iterations to run grabCut
            algorithm for. Defaults to 2.

    Returns:
        np.ndarray: Image ROI array with background of transparent
            (0,0,0) pixels.
    """

    # more advanced segmentation based on colour in cv2?
    #https://www.kaggle.com/code/amulyamanne/image-segmentation-color-clustering/notebook

    # Read/convert image input to np.uint8 np.ndarray (range 0,255).
    if type(img)==str:
        with Image.open(img) as f:
            img_arr = np.array(f, dtype=np.uint8)
    elif img.dtype == np.float32 or img.dtype == np.float64:
        img_arr = np.array(img*255, dtype=np.uint8)
    else:
        img_arr = img

    # Create segment template of (width, height) size of input image.
    segment = np.zeros(img_arr.shape[:2], np.uint8)

    # Unpack bounding box coordinates, set mask values in area corresponding 
    # to area inside bounding box to 1.
    x, y, width, height = bounding_box
    segment[y:(y+height), x:(x+width)] = 1

    # Create templates for bacground and foreground
    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)

    # Run grabCut with bounding box info, prioritise low iter for speed
    try:
        cv2.grabCut(img_arr, segment, bounding_box, background_mdl,
                    foreground_mdl, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:  # unkown error started occuring at index 57 in CAT_00_mixed
        pass

    # Update mask and remove background of image, by multiplying background
    # pixels with 0.
    new_mask = np.where((segment==2)|(segment==0),0,1).astype('uint8')
    cut_img = img_arr*new_mask[:,:,np.newaxis]

    return cut_img


def get_cropped_ROIs(files: list[str], limit: bool = False,
                     verbose: bool = False, save=False) -> dict[str, dict[str, np.ndarray]|list[np.ndarray]]:
    """Take a list of image filenames, run object detection and judge
    suitablility of image for further use. If suitable, use bounding box
    to run grabCut algorithm and return dict of cropped image ROIs and
    excluded images sorted by error type (object detection failure,
    multiple objects found).

    Args:
        files (list[str]): List of iamge filenames.
        limit (bool, optional): Restrict number of files to read.
            Defaults to False.
        verbose (bool, optional): Set amount of info to print. Defaults
            to False.

    Returns:
        dict: Contains image ROI arrary and items where code failed.
    """
    print(f'Preprocessing images...')

    # Slice input data, if limit was passed.
    data_imgs = files[:limit] if limit else files

    # Init container for cropped ROIs, and excluded images.
    img_dict = {'cropped_imgs': {}, 'detect_fail': list(), 'multi_obj': list()}

    # Get overall average size for resizing.
    size = get_avg_size(files)
    # Iterate over data and attempt to extract ROI.
    for imgfile in tqdm(data_imgs):
        # Read and resize image.
        with Image.open(imgfile) as f:
            # Reduce image size to improve runtime: half of own size, or half
            # of average size of current files if ROIs are meant to be saved as
            # matrix later.
            size_to = (int(size[0]/2), int(size[1]/2))
            img = np.array(f.resize(size_to))

        # Run object detection, evaluate output for further processing.
        use_img, _, bbox = get_bbox(img)

        if use_img:
            # Succesfull object detection, continue with grabCut
            cut_image = grabcut_algorithm(img, bbox[0])
            img_dict['cropped_imgs'][imgfile] = cut_image

        else:
            if len(bbox) > 1:
                # Object detection failure, more than 1 cat
                img_dict['multi_obj'].append(imgfile)
            else:
                # Object detection failure, no cat
                img_dict['detect_fail'].append(imgfile)

    if verbose:
        print('Images excluded:', (len(img_dict['detect_fail'])
                                   + len(img_dict['multi_obj'])))
        print('Due to object detection failure:', len(img_dict['detect_fail']))
        print('Due to multiple objects found:', len(img_dict['multi_obj']))
        print(f'Cropped image-ROIs returned from {files[0].split("/")[-2]}:',
              len(img_dict['cropped_imgs']))

    return img_dict


if __name__=='__main__':
    verbosity = True if config['verbose'] == 'True' else False
    # If no explicit directory is passed, default to running for CAT_00 subsets.
    if args.source_directory:
        source_dirs = ([args.source_directory,] if type(args.source_directory)!=list
                       else args.source_directory)
    else:
        source_dirs = [config['CAT_00_solid'], config['CAT_00_mixed']]

    # Get files
    print('Fetching files...')
    for src_dir in source_dirs:
        # one of the configured CAT_00 subsets
        if type(src_dir)==dict:
            with open(src_dir['file_refs'], 'r') as f:
                filenames = f.read().split()
            savefile = src_dir['rois']
        # normal dir/ assume is not cat00
        else:
            filenames = list()
            for root, dirs, files in tqdm(os.walk(src_dir)):
                for fname in sorted(files):
                    if fname.endswith('.jpg'):
                        filenames.append(os.path.join(root, fname))
            savefile = root.split('/')[-1] + '_rois.npy'

        # Get ROIs
        img_dict = get_cropped_ROIs(filenames, limit=LIMIT, verbose=verbosity)
        rois = list(img_dict['cropped_imgs'].values())

        # Save ROIs to file
        np.save(savefile, rois)
        print('ROI arrays saved to', savefile, '\n')

    print('\n--- done ---')
