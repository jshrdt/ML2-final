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
from C4_helper import concat_imgs

warnings.filterwarnings("error", category=ConvergenceWarning)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-new', '--train_new', help='force new training+saving of model',
                    action='store_true', default=False)
parser.add_argument('-col', '--n_colours', help='compress and plot random example',
                    default=33)
parser.add_argument('-ex', '--example', help='compress and plot random example',
                    action='store_true', default=False)
parser.add_argument('-s', '--save', help='save created embeddings',
                    action='store_true', default=False)
parser.add_argument('-lim', '--limit_to', help='limit number of images used',
                    default=False)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

MODELFILE = config['compressor_modelfile']
N_COLOURS = int(args.n_colours)
LIMIT = int(args.limit_to) if args.limit_to != False else -1


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
    if modelfile:
        pickle.dump(model, open(modelfile, "wb"))
        print('Model saved to', modelfile)
    else:
        print('No modelfile to save to specified.')

    return model

### apply ###
def compress_colours(model: KMeans, roi_array3d: np.ndarray) -> np.ndarray:
    """Perform colour compression on an image array using KMeans model
    previously fitted. Return image array of compressed colours.

    Args:
        model (KMeans): KMeans model for colour compression.
        roi_array (np.ndarray): Array of input ROI

    Returns:
        np.ndarray: Roi array with colours replaced by closest KMeans centroid.
    """
    # Reshape image array to 2d, preserving colour dimension.
    roi_array = format_KMeans_input(roi_array3d)

    # Get each pixel colour's closest approximation from model.
    colour_labels = model.predict(roi_array)

    # Rebuild image array with compressed colours, normalise to int for plotting.
    colours_compr = np.uint8(model.cluster_centers_[colour_labels])

    return colours_compr


def get_colour_profile(model: KMeans, img_arrays: np.ndarray) \
        -> tuple[list[np.ndarray], list[dict[tuple, float]]]:
    """Takes matrix of cropped image ROI arrays, performs KMeans colour
    compression, and extracts colour profiles. Returns a list of colour
    compressed image ROI arrays and a list of corresponding colour profile
    dictionaries detailing the frequency of each colour centroid.

    Args:
        model (KMeans): Fitted KMeans model for colour compression.
        img_arrays (np.ndarray): Matrix of image ROIs.

    Returns:
        tuple[list[np.ndarray], list[dict[tuple, float]]]: List of colour
            compressed image ROIs and list of colour profile dicts.
    """
    # If input is a single image, fake a batch dimension.
    if len(img_arrays.shape) < 4: img_arrays = [img_arrays,]

    # Perform KMeans colour compression on image ROIs.
    print('Compressing ROIs....')
    compressed_roi = [compress_colours(model, img_arr) for img_arr in img_arrays]

    # Get corresponding colour profiles for each image:
    # dict of colour centroid: absolute frequency.
    print('Creating colour profiles...')
    col_profiles = [Counter(tuple(map(tuple, roi)))
                    for roi in tqdm(compressed_roi)]

    # Add in colours from centroid lists that were not found in image,
    # to get vectors of same dimensionality later; throw out transparent
    # background pixels (0,0,0).
    for profile in col_profiles:
        for col in model.cluster_centers_:
            profile[tuple(np.uint8(col))] = profile.get(tuple(np.uint8(col)), 0)
        del profile[(0, 0, 0)]

    # Normalise frequencies by roi size to get relative. Drop arrays where
    # no non-transparent pixels remained (additional grabCut fail)
    profiles_norm = [{col: count/sum(profile.values()) 
                      for col, count in profile.items()

                      }
                     for profile in col_profiles
                     if sum(profile.values())>0
                     
                     ]
    
    print([len(c) for c in profiles_norm])

    # Sort profiles by colour centroids to have their order align across
    # dimensions.
    profiles_sort = [dict(sorted(profile.items())) for profile in profiles_norm]

    return compressed_roi, profiles_sort


def vectorize_colours(c_profiles: list[dict[tuple, float]]) -> list[np.ndarray]:
    """Transform colour profiles to embeddings vectors  of N_COLOURS-1 dims."""
    col_embeddings = [np.array(list(profile.values())) for profile in c_profiles]

    return col_embeddings


def apply_colour_compression(model: KMeans, rois: np.ndarray) -> list[np.ndarray]:
    """High level function, transform image ROI matrix to colour embeddings.

    Args:
        model (KMeans): Fitted KMeans model for colour compression. 
        rois (np.ndarray): Matrix of image ROI arrays.

    Returns:
        list[np.ndarray]: List of colour embeddings from image ROIs.
    """
    # Get colour profiles
    _, c_profiles = get_colour_profile(model, rois)

    # transform to colour embedding vectors
    colour_embeddings = vectorize_colours(c_profiles)

    return colour_embeddings

### move to example.py ###
def plot_example(model, roi):
    """Take a single array from a cropped image ROI, plot the input, and
    its colour compressed version plus colour profile decompositon.
    """

    # create plot for: cropped ROI vs colour compressed ROI ...
    compressed_roi, profile = get_colour_profile(model, roi)
    compressed_img = np.uint8(compressed_roi[0].reshape(roi.shape[0],
                                                        roi.shape[1],
                                                        -1))

    fig1, ax1 = plt.subplots(1,2)
    ax1[0].imshow(roi)
    ax1[1].imshow(compressed_img)
    ax1[0].set_title('Image post grabCut')
    ax1[1].set_title('Post KMeans colour compression')

    #... and for colour profile
    profile = {col: freq*100 for col, freq in profile[0].items()
               if freq>0}
    df = pd.DataFrame(profile.items(), columns=['colours', 'count'],
                      index = [str(col) for col in profile.keys()]
                      #'#%02x%02x%02x' % co
                      )
    # drop some colours with very low values from the plot, sort by frequency
    df = df.drop(df[df['count'] < 0.01].index).sort_values(by='count', ascending=False)
    mycols = ['#%02x%02x%02x' % col for col in df['colours']]

    fig2 = plt.figure()
    fig2.tight_layout
    ax2 = fig2.add_subplot(111)

    df.plot.barh(color={'count': mycols}, ax=ax2)
    ax2.set_title(f'Compressed colour profile (n={N_COLOURS-1}) of ROI (freq>0,01%)')
    ax2.legend(labels=['Relative frequency of colour in ROI in percent'],
               loc='upper right')

    plt.show()
    print('--- end ---')

### move to example.py ###
def display_palette(model):
    """Display colour palette used in KMeans model."""
    gold_palette = np.array([np.uint8(col) for col in model.cluster_centers_
                         if sum(np.uint8(col)>0)])
    plt.imshow((gold_palette/255).reshape(1, len(gold_palette), 3))
    plt.show()


if __name__=='__main__':
    if os.path.isfile(MODELFILE) and args.train_new==False:
        # Load model for colour compression
        kmeans_model = pickle.load(open(MODELFILE, "rb"))
    else:
        print('Training new model...')
        # train new kmeans colour compression
        gold_rois = np.load(config['gold_img_ROIs_file'])
        kmeans_model = train_compressor(gold_rois, colours=N_COLOURS,
                                        modelfile=MODELFILE)
        if args.save or os.path.isfile(MODELFILE)==False:
            gold_c_embeds = apply_colour_compression(kmeans_model, gold_rois)
            np.save('./cropped/'+config['gold_embeds'], gold_c_embeds)
            print('Gold roi colour embeddings saved to ./cropped/'+config['gold_embeds'])

    # Load already croppped test image rois for compression
    gen_rois = np.load(config['gen_ROIs'])[:LIMIT]
    print(len(gen_rois))

    if args.example:
        #display_palette(kmeans_model)
        plot_example(kmeans_model, gen_rois[random.randint(0, len(gen_rois)-1)])

    else:
        print('\nApplying colour compression')
        c_embeds = apply_colour_compression(kmeans_model, gen_rois)

        # ? save to numpy
        if config['gen_embeds'] and args.save:
            np.save(config['gen_embeds'], c_embeds)
            print('Colour embeddings saved to', config['gen_embeds'])

    print('\n--- done ---')

