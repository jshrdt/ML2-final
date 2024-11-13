
### adapted from ###
# https://scikit-learn.org/1.5/auto_examples/cluster/plot_color_quantization.html
# Authors: Robert Layton, Olivier Grisel, Mathieu Blondel
# License: BSD 3 clause
####################

import argparse
import json
import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import pickle

from C4_helper import concat_imgs, save_kmeans_model, get_rois

warnings.filterwarnings('error', category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-new', '--train_new', help='force new training+saving of model',
                    action='store_true', default=False)
parser.add_argument('-col', '--n_colours', help='compress and plot random example',
                    default=32)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

N_COLOURS = int(args.n_colours) +1  # to account for (0,0,0) exclusion later


def get_compress_model(gold_dir: dict = None) -> KMeans:
    """High-level function to load/fit model for KMeans colour compression."""
    # Try loading model from file...
    if os.path.isfile(gold_dir['compressor_modelfile']) and args.train_new==False:
        # Load model for colour compression
        model = pickle.load(open(gold_dir['compressor_modelfile'], "rb"))
        print('Model for colour compression loaded from', gold_dir['compressor_modelfile'])

    elif gold_dir:
        print('Fitting new model for colour compression on', gold_dir['rois'])
        # or fit a new model.
        # Load data for fitting & fit model; save model if modelfile passed.
        gold_rois = get_rois(gold_dir, verbose=config['verbose'], save=True)

        model = train_compressor(gold_rois, colours=N_COLOURS,
                                 modelfile=gold_dir['compressor_modelfile'])
    else:
        raise ValueError('Unable to laod or train new model')

    return model


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
    # Concatenate image arrays to one big matrix, approximate square dimensions.
    gold_matrix = concat_imgs(gold_img_arrays)
    # Flatten data to 2d matrix, preserving colour dimensions.
    gold_matrix_norm = format_KMeans_input(gold_matrix)

    try:
        # Fit model on subsample of matrix.
        matrix_sample = shuffle(gold_matrix_norm, random_state=0,
                                n_samples=samples)
        model = KMeans(n_clusters=colours, random_state=0).fit(matrix_sample)

    except ConvergenceWarning:
        return 0

    # Save model to file.
    if modelfile:
        save_kmeans_model(model, modelfile)

    return model


def format_KMeans_input(array3d: np.ndarray) -> np.ndarray:
    """Reshape 3d of image to 2d array, preserve colour dimension."""
    width, height, col_dims = array3d.shape
    array2d = np.reshape(array3d, (width * height, col_dims))

    return array2d


def compress_roi(roi: np.ndarray, model: KMeans) -> np.ndarray:
    """Perform colour compression on an image array using previously
    fitted KMeans model. Return 2d image array of compressed colours.

    Args:
        model (KMeans): KMeans model for colour compression.
        roi (np.ndarray): 3dArray of input ROI.

    Returns:
        np.ndarray: ROI array with colours replaced by closest KMeans centroid.
    """
    # Reshape image array to 2d, preserving colour dimension.
    roi_array = format_KMeans_input(roi)

    # Get each pixel colour's closest approximation from model.
    colour_labels = model.predict(roi_array)

    # Rebuild image array with compressed colours, normalise to int for plotting.
    compressed_roi = np.uint8(model.cluster_centers_[colour_labels])

    return compressed_roi


def get_colour_profile(compressed_roi: np.ndarray, palette: list[tuple]) \
        -> dict[tuple, float]:
    """Takes list of compressed image ROI arrays, extracts and returns
    colour profiles: a list of dictionaries detailing the frequency of
    each colour from the compression palette per image.

    Args:
        compressed_rois (list[np.ndarray]): List of colour compressed ROI 
            arrays.
        palette (list[tuple]): List of unique colour centroids from colour
            compression model as RGB tuples.

    Returns:
        dict[tuple, float]]: Relative frequency distribution of colour
            centroids across input array.
    """
    # Get raw colour profile from input array as a dict mapping each unique
    # colour centroid int the image to its absolute frequency therein.
    colour_profile = Counter(tuple(map(tuple, compressed_roi)))

    # Add in colours from centroid list that did not occur in image array
    # to even out dimensions.
    for col in palette:
        colour_profile[col] = colour_profile.get(col, 0)
    del colour_profile[(0, 0, 0)]

    # Normalise counts by ROI size to get relative frequencies.
    profile_norm = {col: count/sum(colour_profile.values())
                    if sum(colour_profile.values()) > 0 else 0.0  # avoid zero division error
                    for col, count in colour_profile.items()}

    # Sort profiles by colour centroids' red value to have their dimensions
    # align across input samples.
    profile_sorted = dict(sorted(profile_norm.items()))

    return profile_sorted


def plot_example(model: KMeans, roi: np.ndarray):
    """Take a single array from a cropped image ROI; plot the input, its
    colour compressed version with the model's colour palette (1), plus its
    colour profile decompositon (2).
    """
    # 1: Create plot for: cropped ROI vs colour compressed ROI + colour palette...
    compr_roi = compress_roi(roi, model)
    # Extract palette from model and sort by shades.Exlude (0,0,0) which is
    # also excluded from the vectorised colour profiles during clustering later.
    palette = sorted([tuple(np.uint8(col)) for col in model.cluster_centers_
               if sum(np.uint8(col)>0)], reverse=True)
    profile = get_colour_profile(compr_roi, palette)
    compr_img = np.uint8(compr_roi.reshape(roi.shape[0], roi.shape[1], 3))
    # Reshape to horizontal bar/image, sorted by shade.
    colour_palette = np.uint8(palette).reshape(1, N_COLOURS-1, 3)
    # Create plots
    fig = plt.figure()
    gs = fig.add_gridspec(2,2, height_ratios=[5, 0.5])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])
    # Plot
    ax1.imshow(roi)
    ax2.imshow(compr_img)
    ax3.imshow(colour_palette)
    # Add info
    ax1.set_title('Image post grabCut (input)')
    ax2.set_title('Post KMeans colour compression')
    ax3.set_title('Complete spectrum of colour centroids in compression model')

    #2: ... and similarly for the colour profile.
    # Rescale relative frequency to range 0, 100.
    profile = {col: freq*100 for col, freq in profile.items() if freq>0}
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


if __name__=='__main__':
    gold_dir = config['CAT_00_solid']
    test_dir = config['CAT_01']

    # run code on only one random image from test rois to plot:
    # 1.1) test image ROI, 1.2) test image ROI post colour compression,
    # 1.3) colour palette of model, 2) test image colour profile graph.
    kmeans_model = get_compress_model(gold_dir)
    test_roi = get_rois(test_dir, limit=1, is_ex=True)[0]
    plot_example(kmeans_model, test_roi)

    print('\n--- done ---')
