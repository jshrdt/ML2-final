# loader
import os
import numpy as np
import random

import pickle

from process_imgs import get_cropped_ROIs


def concat_imgs(img_arrays: list) -> np.ndarray:
    """Concat list of image arrays to matrix to facilitate fitting/plots."""
    step = int(len(img_arrays)**0.5+len(img_arrays)**0.2)
    img_matrix = np.concatenate([np.concatenate(img_arrays[i-step:i], axis=0)
                                 for i in range(step, len(img_arrays), step)],
                                axis=1)
    return img_matrix


def save_kmeans_model(model, modelfile: str = None):
    """Save sklearn cluster KMeans model to file."""
    if modelfile:
        pickle.dump(model, open(modelfile, "wb"))
        print('Model saved to', modelfile, '\n')
    else:
        print('No modelfile to save to specified.')


def get_rois(config_dir: dict, limit: bool = False, verbose: bool = False,
             save: bool = False, is_ex: bool = False) -> list[np.ndarray]:
    """Helper function to dynamically load/Create iamge ROIs across files.

    Args:
        config_dir (dict): Config dict for data.
        limit (bool, optional): Reduce number of items. Defaults to False.
        verbose (bool, optional): Set side effect behaviour when creating
            ROI arrays from scratch. Defaults to False.
        save (bool, optional): Set whether to save ROI arrays to file,
            affects resizing. Defaults to False.
        is_ex (bool, optional): Configures code to return only a single
            random ROI within range. Defaults to False.

    Returns:
        list[np.ndarray]: _description_
    """
    if os.path.isfile(config_dir['rois']):
        print('Loading ROI arrays from...', config_dir['rois'])
        rois = np.load(config_dir['rois'])
        # If example: get 1 random example in range
        if is_ex:
            np.random.default_rng().shuffle(rois)
        if limit: rois = rois[:limit]

    # else:
    #     pass  # ROIs are supplied
    else:
        if config_dir.get('file_refs', 0):
            print('Creating ROI arrays using', config_dir['file_refs'])
            # Get filenames
            with open(config_dir['file_refs'], 'r') as f:
                files = f.read().split()
        else:
            files = list()
            for root, dirs, filenames in os.walk(config_dir['imgs_dir']):
                for fname in sorted(filenames):
                    if fname.endswith('.jpg'):
                        files.append(os.path.join(root, fname))
        if is_ex:
            random.shuffle(files)
        # Get image ROI arrays.
        img_dict = get_cropped_ROIs(files, limit=limit, verbose=verbose)
        rois = list(img_dict['cropped_imgs'].values())

        # Save ROI arrays to file.
        if is_ex==False and (save or os.path.isfile(config_dir['rois'])==False):
            np.save(config_dir['rois'], rois)
            print('ROI arrays saved to', config_dir['rois'], '\n')

    return rois

## Used to create CAT_00 subsets; no longer in use ##
# def create_gold_refs(data_dir):
#     with open(data_dir['file_refs'], 'w') as f:
#         for root, dirs, files in os.walk(data_dir['imgs_dir']):
#             for fname in sorted(files):
#                 if fname.endswith('.jpg'):
#                     f.write(f"{'./cats/CAT_00/'+fname}\n")

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
