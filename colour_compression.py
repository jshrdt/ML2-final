# colour compression 

### adapted from ###
# https://scikit-learn.org/1.5/auto_examples/cluster/plot_color_quantization.html
# Authors: Robert Layton, Olivier Grisel, Mathieu Blondel
# License: BSD 3 clause
####################

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning

### edited by moi:
import argparse
import json
import warnings
import pandas as pd
import pickle

from collections import Counter
import random 
from tqdm import tqdm
import os
from C4_helper import concat_imgs, save_kmeans_model

warnings.filterwarnings("error", category=ConvergenceWarning)


parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-new', '--train_new', help='force new training+saving of model',
                    action='store_true', default=False)
parser.add_argument('-col', '--n_colours', help='compress and plot random example',
                    default=33)
parser.add_argument('-ex', '--example', help='compress and plot random example',
                    action='store_true', default=False)
parser.add_argument('-s', '--save', help='create & save colour embeddings',
                    action='store_true', default=False)
parser.add_argument('-lim', '--limit', help='limit number of images used',
                    default=0)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

MODELFILE = config['compressor_modelfile']
N_COLOURS = int(args.n_colours)
LIMIT = int(args.limit) if args.limit != "False" else False

def get_model(modelfile=None) -> KMeans:
    """High-level function to load/fit model for KMeans colour compression."""
    # Try loading model from file...
    if modelfile and os.path.isfile(modelfile) and args.train_new==False:
        # Load model for colour compression
        model = pickle.load(open(MODELFILE, "rb"))
    else:
        # or fit a new model.
        # Load data for fitting & fit model; save model if modelfile passed.
        gold_rois = np.load(config['gold_ROIs'])
        model = train_compressor(gold_rois, colours=N_COLOURS,
                                 modelfile=modelfile)
        # optional: get and save gold embeds created; needed for clustering.
        if args.save or os.path.isfile(config['gold_embeds'])==False:
             # Get colour profiles & transform to colour embedding vectors
            _, gold_profiles = get_colour_profile(model, gold_rois)
            gold_c_embeds = vectorize_colours(gold_profiles)
            # Save to file
            np.save(config['gold_embeds'], gold_c_embeds)
            print('Gold roi colour embeddings saved to', config['gold_embeds'])

    return model

def format_KMeans_input(array3d: np.ndarray) -> np.ndarray:
    """Reshape 3d of image to 2d array, preserve colour dimension."""
    width, height, col_dims = array3d.shape
    array2d = np.reshape(array3d, (width * height, col_dims))

    return array2d

### train ###
def train_compressor(gold_img_arrays: np.ndarray, colours: int = N_COLOURS,
                     samples: int = 10_000,  modelfile: str = None) -> KMeans:
    """Fit a KMeans model on a matrix of concatenated high quality image
    ROIs to perform colour compression. Fit on variable number of colours
    and random samples. Optionally save trained model to file.

    Args:
        gold_img_arrays (np.ndarray): List of ROI arrays as training data.
        colours (int, optional): Maximum number of colours to compress to.
            Defaults to N_COLOURS=33.
        samples (int, optional): Number of samples to pick from gold image
            matrix. Defaults to 10_000.
        modelfile (str, optional): Filename to save model to. Defaults to None.

    Returns:
        KMeans: KMeans model for colour compression.
    """
    print('Formatting data...')
    # Concatenate image arrays to one big matrix, approximate square dimensions.
    gold_matrix = concat_imgs(gold_img_arrays)

    # Flatten data to 2d matrix, preserving colour dimensions.
    gold_matrix_norm = format_KMeans_input(gold_matrix)

    try:
        # Fit model on subsample of matrix.
        print('Fitting model...')
        matrix_sample = shuffle(gold_matrix_norm, random_state=0,
                                n_samples=samples)
        model = KMeans(n_clusters=colours, random_state=0).fit(matrix_sample)

    except ConvergenceWarning:
        return 0

    # Save model to file.
    if (modelfile and os.path.isfile(modelfile)==False) or args.train_new:
        raise ValueError('shoudl not save')
        save_kmeans_model(model, modelfile)

    return model


### apply ###
def compress_colours(model: KMeans, roi_array3d: np.ndarray) -> np.ndarray:
    """Perform colour compression on an image array using previously
    fitted KMeans model. Return 2d image array of compressed colours.

    Args:
        model (KMeans): KMeans model for colour compression.
        roi_array3d (np.ndarray): Array of input ROI.

    Returns:
        np.ndarray: Roi array with colours replaced by closest KMeans centroid.
    """
    # Reshape image array to 2d, preserving colour dimension.
    roi_array = format_KMeans_input(roi_array3d)

    # Get each pixel colour's closest approximation from model.
    colour_labels = model.predict(roi_array)

    # Rebuild image array with compressed colours, normalise to int for plotting.
    compressed_roi_array = np.uint8(model.cluster_centers_[colour_labels])

    return compressed_roi_array


def get_colour_profile(model: KMeans, compressed_rois: list[np.ndarray]) \
        -> list[dict[tuple, float]]:
    """Takes list of compressed image ROI arrays, extracts and returns
    colour profiles: a list of dictionaries detailing the frequency of
    each colour from the compression palette per image.

    Args:
        model (KMeans): Fitted KMeans model for colour compression.
        compressed_rois (list[np.ndarray]): List of colour compressed ROI 
            arrays.

    Returns:
        list[dict[tuple, float]]]: List of colour profile dicts for each
            compressed roi from input.
    """
    print('Creating colour profiles...')
    # Get corresponding colour profiles for each image:
    # dict of colour centroid: absolute frequency.
    col_profiles = [Counter(tuple(map(tuple, roi))) for roi in tqdm(compressed_rois)]

    # Add in colours from centroid lists that were not found in image and
    # throw out background pixels marked by grabCut (0,0,0).
    for profile in col_profiles:
        for col in model.cluster_centers_:
            profile[tuple(np.uint8(col))] = profile.get(tuple(np.uint8(col)), 0)
        del profile[(0, 0, 0)]

    # Normalise frequencies by roi size to get relative. Preserve arrays where
    # no non-transparent pixels remained (grabCut fail) to avoid issues with
    # indexing items.
    profiles_norm = [{col: count/sum(profile.values())
                      if sum(profile.values()) > 0 else 0.0
                      for col, count in profile.items()}
                     for profile in col_profiles]

    # Sort profiles by colour centroids to have their order align across
    # dimensions.
    profiles_sort = [dict(sorted(profile.items())) for profile in profiles_norm]

    return profiles_sort


def vectorize_colours(c_profiles: list[dict[tuple, float]]) -> list[np.ndarray]:
    """Transform colour profiles to embeddings vectors  of N_COLOURS-1 dims."""
    print('Vectorising profiles...')
    col_embeddings = [np.array(list(profile.values())) for profile in c_profiles]

    return col_embeddings


def plot_example(model: KMeans, roi: np.ndarray):
    """Take a single array from a cropped image ROI; plot the input, its
    colour compressed version with the model's colour palette (1), plus its
    colour profile decompositon (2).
    """
    # 1: Create plot for: cropped ROI vs colour compressed ROI + colour palette...
    compressed_roi = compress_colours(model, roi)
    profile = get_colour_profile(model, [compressed_roi,]) # fake dimensiosn
    compressed_img = np.uint8(compressed_roi.reshape(roi.shape[0], roi.shape[1],
                                                     3))
    # Extract full colour palette from model, reshape to horizontal bar/image.
    colour_palette = np.uint8(sorted([tuple(col) for col in model.cluster_centers_
                                      if sum(np.uint8(col)>0)], reverse=True))
    colour_palette = (colour_palette / 255).reshape(1, len(colour_palette), 3)

    # Create plots
    fig = plt.figure()
    gs = fig.add_gridspec(2,2, height_ratios=[5, 0.5])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])
    # Plot
    ax1.imshow(roi)
    ax2.imshow(compressed_img)
    ax3.imshow(colour_palette)
    # Add info
    ax1.set_title('Image post grabCut (input)')
    ax2.set_title('Post KMeans colour compression')
    ax3.set_title('Complete spectrum of colour centroids in compression model')

    #2: ... and similarly for the colour profile.
    profile = {col: freq*100 for col, freq in profile[0].items() if freq>0}
    df = pd.DataFrame(profile.items(), columns=['colours', 'count'],
                      index = [str(col) for col in profile.keys()])
    # drop very rare colours (<0.01%) from the plot, sort colours by frequency.
    df = df.drop(df[df['count']<0.01].index).sort_values(by='count',
                                                         ascending=False)
    # Set colours of bars to align with colour from image.
    mycols = ['#%02x%02x%02x' % col for col in df['colours']]

    # Create plots
    fig2 = plt.figure(figsize=(9,6))
    a0 = fig2.add_subplot(111)
    # Plot
    df.plot.barh(color={'count': mycols}, ax=a0)
    # Add info
    a0.set_title(f'Compressed colour profile (n={N_COLOURS-1}) of ROI (freq>0,01%)')
    a0.set_xlabel('Prevalence of colours in compressed ROI (%)')
    a0.set_ylabel('Colour as RGB code values')
    a0.legend(labels=['Relative frequency of colour'], loc='upper right')

    plt.show()


def get_colour_embeddings(test_rois: np.ndarray, save: bool = False) -> list[np.ndarray]:
    """High level function, load/fit KMeans model for colour compression,
    load image ROIs from test data, transform and return their colour
    embeddings. Optionally save embeddings to file.

    Args:
        test_rois (np.ndarray): _description_
        save (bool, optional): Set whether to save colour embeddings to
            file. Defaults to False.

    Returns:
        list[np.ndarray]: _description_
    """
    # Get model (try to load from config file, else try to fit new model
    # with gold_ROIs)
    kmeans_model = get_model(MODELFILE)

    # Perform KMeans colour compression on image ROIs.
    print('Compressing ROIs....')
    compressed_rois = [compress_colours(kmeans_model, img_arr)
                       for img_arr in test_rois]
    # Get colour profiles
    colour_profiles = get_colour_profile(kmeans_model, compressed_rois)

    # transform to colour embedding vectors
    colour_embeddings = vectorize_colours(colour_profiles)

    # optional: save embeds to numpy file
    if config['gen_embeds'] and save:
        np.save(config['gen_embeds'], colour_embeddings)
        print('Colour embeddings saved to', config['gen_embeds'])

    return colour_embeddings


if __name__=='__main__':
    if args.example:
        # run code on only one random image from test rois to plot: 
        # 1.1) test image ROI, 1.2) test image ROI post colour compression,
        # 1.3) colour palette of model, 2) test image colour profile graph.
        kmeans_model = get_model(MODELFILE)
        if os.path.isfile(config['gen_ROIs']):
            gen_rois = np.load(config['gen_ROIs'])
            test_roi = gen_rois[random.randint(0, len(gen_rois)-1)]
        plot_example(kmeans_model, test_roi)

    else:
        if os.path.isfile(config['gen_ROIs']):
            # Load pre-processed (cropped) test image rois.
            test_rois = np.load(config['gen_ROIs'])
        else:
            # try to create?
            raise FileNotFoundError('No image ROIs found at', config['gen_ROIs'])
            # ....
        # Create colour embeddings from config npy file; if --save: save
        # to config embeds file.
        if LIMIT: test_rois = test_rois[:LIMIT]

        col_embeds = get_colour_embeddings(test_rois, args.save)
    print('\n--- done ---')

