# loader
import os
import numpy as np
import random

import pickle

from preprocess_imgs import get_cropped_ROIs



def concat_imgs(img_arrays: list) -> np.ndarray:
    """Concat list of image arrays to matrix to facilitate fitting/plots."""
    step = int(len(img_arrays)**0.5+len(img_arrays)**0.2)
    img_matrix = np.concatenate([np.concatenate(img_arrays[i-step:i], axis=0)
                                 for i in range(step, len(img_arrays), step)],
                                axis=1)
    return img_matrix

def save_kmeans_model(model, modelfile=None):
    if modelfile:
        pickle.dump(model, open(modelfile, "wb"))
        print('Model saved to', modelfile, '\n')
    else:
        print('No modelfile to save to specified.')


### write func to load gold rois only

def get_rois(config_dir, limit=False, verbose=False, save=False, is_ex=False):
    if os.path.isfile(config_dir['rois']):
        print('Loading ROI arrays from...', config_dir['rois'])
        rois = np.load(config_dir['rois'])
        if is_ex:
            np.random.default_rng().shuffle(rois)

        if limit: rois = rois[:limit]
    else:
        print('Creating ROI arrays using file refs from', config_dir['file_refs'])
        with open(config_dir['file_refs'], 'r') as f:
            files = f.read().split()

        if is_ex:
            random.shuffle(files)


        img_dict = get_cropped_ROIs(files, limit=limit, save=save,
                                    verbose=verbose)

        rois = list(img_dict['cropped_imgs'].values())

        if is_ex==False and (save or os.path.isfile(config_dir['rois'])==False):
            np.save(config_dir['rois'], rois)
            print('ROI arrays saved to', config_dir['rois'], '\n')

    return rois

def create_gold_refs(data_dir):
    with open(data_dir['file_refs'], 'w') as f:
        for root, dirs, files in os.walk(data_dir['imgs_dir']):
            for fname in sorted(files):
                if fname.endswith('.jpg'):
                    f.write(f"{'./cats/CAT_00/'+fname}\n")

# def gold_rois_from_imgs():
#     gold_arrs = list()
#     for root, dirs, files in os.walk('./cropped/imgs'):
#         for fname in sorted(files):
#             if fname.endswith('.jpg'):
#                 with Image.open('./cropped/imgs/'+fname) as f:
#                     gold_arrs.append(np.array(f))
#     return gold_arrs

if __name__=='__main__':
    pass
