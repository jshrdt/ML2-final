import argparse
import json
import os

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
parser.add_argument('-srcdir', '--source_directory',
                    help='directory to run preprocessing on',
                    default=False)
parser.add_argument('-lim', '--limit',
                    help='limit number of images used from dir for code testing',
                    default=False)
args, unk = parser.parse_known_args()
config = json.load(open(args.config))

verbose = True if config['verbose'] == 'True' else False
source_dir = args.source_directory if args.source_directory else config['toy_dir']


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
                     limit: bool = False) -> dict:
    """Read and resize images, run object detetion and judge suitablility
    of image for further use. If suitable, use bounding box to run grabCut
    algorithm and return dict of image ROIs and pruned images."""
    print('Preprocessing images...')

    size_to = (int(data.avg_size[0]/2), int(data.avg_size[1]/2))
    data_imgs = data.imgfiles[:limit] if limit else data.imgfiles

    # compressed_ROIs = list()
    img_dict = {'cropped_imgs': {},
                'detect_fail': list(),
                'multi_obj': list()}

    for imgfile in tqdm(data_imgs):
        # Read and resize image to half its size to improve runtime
        with Image.open(imgfile) as f:
            img = np.array(f.resize(size_to))

        # Run object detection, evaluate output for further processing.
        use_img, _, bbox = get_bbox(img)
        ## TBD work with labels and conf thresholds?
        if use_img:
            cut_image = grabcut_algorithm(img, bbox[0])
            img_dict['cropped_imgs'][imgfile] = cut_image
            #img_dict['cropped_imgs']['filenames'].append(imgfile)

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
        print(f'Cropped image-ROIs returned from {data.data_dir} :',
              len(img_dict['cropped_imgs']))

    return img_dict

def create_gold_refs():
    with open('./cropped/gold_file_refs.txt', 'w') as f:
        for root, dirs, files in os.walk('./cropped/imgs'):
            for fname in sorted(files):
                if fname.endswith('.jpg'):
                    f.write(f'{fname}\n')
                    
def gold_rois_from_imgs():
    for root, dirs, files in os.walk('./cropped/imgs'):
        for fname in sorted(files):
            if fname.endswith('.jpg'):
                with Image.open('./cropped/imgs/'+fname) as f:
                    gold_arrs.append(np.array(f))

#### save as img
# from PIL import Image
# if not os.path.isdir('./cropped/imgs'):
#         os.makedirs('./cropped/imgs')

# for i, img_arr in enumerate(img_dict['cropped_imgs']['img_arrs']):
#     im = Image.fromarray(img_arr)
#     fname = img_dict['cropped_imgs']['filenames'][i].split('/')[-1]
#     im.save('./cropped/imgs/' + fname)


###

    # if not os.path.isdir('./cropped'):
    #     os.makedirs('./cropped_dev')
    # # save devs
    # np.save('./cropped/' + img_ROIs_file, devs)
    # np.save('./cropped/' + img_gold_ROIs_files, golds)
    
    ## visualise 
#     import matplotlib.pyplot as plt
# from PIL import Image
# plt.imshow(Image.open('./cropped/imgs/'+data.gold_imgs[0].split('/')[-1]))


if __name__=='__main__':
    # preprocess cat00, all imgs togehter
    # then sort into gold/normal, and save as 2 numpy files
    # fetch files
    data = DataLoader(source_dir)
    # preprocess images
    img_dict = get_cropped_ROIs(data, verbose=config['verbose'], limit=False) 
    # run this far to check processing speed/preprocessing success rate

    # filter out gold images
    gold_arrs, gen_arrs = list(), list()
    if 'CAT_00' in source_dir:
        with open('./cropped/gold_file_refs.txt', 'r') as f:
            gold_file_refs = f.read().split()
        for fname, img_arr in img_dict['cropped_imgs'].items():
            if fname.split('/')[-1] in gold_file_refs:
                gold_arrs.append(img_arr)
            else:
                gen_arrs.append(img_arr)

        # if config['gold_img_ROIs_file'] and not os.path.isfile(config['gold_img_ROIs_file']):
        #     np.save(config['gold_img_ROIs_file'], gold_arrs)
    else:
        gen_arrs = list(img_dict['cropped_imgs'].values())

    print(len(gen_arrs), len(gold_arrs))

    # save (general) iamge rois as numpy matrix
    if config['gen_img_ROIs_file']:
        np.save('./cropped/'+config['gen_img_ROIs_file'], gen_arrs)