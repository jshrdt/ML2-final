# colour compression 

### adapted from ###
# https://scikit-learn.org/1.5/auto_examples/cluster/plot_color_quantization.html
# Authors: Robert Layton, Olivier Grisel, Mathieu Blondel
# License: BSD 3 clause
####################

# import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.cluster import KMeans
#from sklearn.datasets import load_sample_image
# from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning

### edited by moi:
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("error", category=ConvergenceWarning)
import pandas as pd
import pickle

from collections import Counter
import random 
from tqdm import tqdm
import os


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

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

MODELFILE = config['compressor_modelfile']
N_COLOURS = int(args.n_colours)


### train ###
def train_compressor(img_matrix, colours=N_COLOURS, samples=10_000, modelfile=None):
    print('Formatting data...')
    # Format data as 2d matrix, preserving colour dimensions
    width, height, col_dims = img_matrix.shape
    img_matrix_norm = np.reshape(img_matrix, (width * height, col_dims))

    try:
        print('Fitting model...')
        image_matrix_sample = shuffle(img_matrix_norm, random_state=0,
                                      n_samples=samples)
        model = KMeans(n_clusters=colours,
                       random_state=0).fit(image_matrix_sample)
    except ConvergenceWarning:
        return 0

    if modelfile:
        pickle.dump(model, open(modelfile, "wb"))
        print('Model saved to', modelfile)
    else:
        print('No modelfile to save to specified.')

    return model

### apply ###
def compress_colours(model: KMeans, img_array: np.ndarray):
    width, height, col_dims = img_array.shape
    img_array = np.reshape(img_array, (width * height, col_dims))

    # Get closest colours for all ROI pixels
    colour_labels = model.predict(img_array)

    # rebuild image with compressed colours
    colours_compr = np.uint8(model.cluster_centers_[colour_labels])
    # reshape image to previous form & normalise to int values for plotting

    return colours_compr


# def recreate_image(cluster_centers, labels, w, h): # imported & edited from source
#     """Rebuild image with compressed colours, normalised to int values"""
#     return np.uint8(cluster_centers[labels].reshape(w, h, -1))


def get_colour_profile(model: KMeans, arrs: np.ndarray):
    if len(arrs.shape)<4: arrs = [arrs,]

    print('Compressing ROIs....')
    compressed_roi = [compress_colours(model, img_arr) for img_arr in arrs]

    print('Creating colour profiles...')
    c_profiles = [Counter(tuple(map(tuple, roi))) for roi in tqdm(compressed_roi)]
    ## sort, normalise by roi size, throw ot (0,0,0)
    c_profiles_norm = [{col: count/sum(profile.values())
                                    for col, count in profile.items()
                                    if sum(col)>0}
                       for profile in c_profiles 
                       #if sum(profile.values())>0
                       ]
    # add non existent colours bac in, to have same len of vectors later
    for profile in c_profiles_norm:
        for col in model.cluster_centers_:
            col = tuple(np.uint8(col))
            if col not in set(profile.keys()) and sum(col)>0:
                profile[col] = 0

    c_profiles_sorted = [dict(sorted(profile.items())) for profile in c_profiles_norm]

    return compressed_roi, c_profiles_sorted


def vectorize_colours(c_profiles):
    col_embeddings = [np.array(list(profile.values())) for profile in c_profiles]

    return col_embeddings


def apply_colour_compression(model, rois):
    # Get colour profiles
    _, c_profiles = get_colour_profile(model, rois)

    # transform to colour embedding vectors
    colour_embeddings = vectorize_colours(c_profiles)

    return colour_embeddings

### move to example.py ###
def plot_example(model, roi):

    # create plot for: cropped ROI vs colour compressed ROI ...
    compressed_roi, profile = get_colour_profile(model, roi)
    compressed_img = np.uint8(compressed_roi[0].reshape(roi.shape[0], roi.shape[1], -1))

    fig1, ax1 = plt.subplots(1,2)
    ax1[0].imshow(roi)
    ax1[1].imshow(compressed_img)
    ax1[0].set_title('Image post grabCut')
    ax1[1].set_title('Post KMeans colour compression')

    #... and for colour profile
    profile = {col: freq*100 for col, freq in profile[0].items()
               if freq>0}
    df = pd.DataFrame(profile.items(), columns=['colours', 'count'],
                      index = ['#%02x%02x%02x' % col for col in profile.keys()])
    df = df.drop(df[df['count'] < 0.01].index)
    mycols = ['#%02x%02x%02x' % col for col in df['colours']]

    fig2 = plt.figure()
    fig2.set_figheight(8)
    ax2 = fig2.add_subplot(111)
    df.sort_values(by='count', ascending=False).plot.barh(
        color={'count': mycols}, ax=ax2)
    ax2.set_title(f'Compressed colour profile (n={N_COLOURS-1}) of ROI (freq>0,01%)')
    ax2.legend(labels=['Relative frequency of colour in ROI in percent'],
               loc='upper right')

    plt.show()
    print('--- end ---')

### move to example.py ###
def display_palette(model):
    gold_palette = np.array([np.uint8(col) for col in model.cluster_centers_
                         if sum(np.uint8(col)>0)])
    plt.imshow((gold_palette/255).reshape(1, len(gold_palette), 3))

if __name__=='__main__':
    if os.path.isfile(MODELFILE) and args.train_new==False:
        # Load model for colour compression
        kmeans_model = pickle.load(open(MODELFILE, "rb"))
    else:
        print('Training new model...')
        # train new kmeans colour compression
        gold_rois = np.load(config['gold_img_ROIs_file'])
        step = int(len(gold_rois)**0.5+len(gold_rois)**0.1)
        gold_matrix = np.concatenate([np.concatenate(gold_rois[i-step:i], axis=0)
                                    for i in range(step, len(gold_rois), step)],
                                    axis=1)
        kmeans_model = train_compressor(gold_matrix, colours=N_COLOURS,
                                        modelfile=MODELFILE)
        if args.save or os.path.isfile(MODELFILE)==False:
            gold_c_embeds = apply_colour_compression(kmeans_model, gold_rois)
            np.save('./cropped/'+config['gold_embeds'], gold_c_embeds)
            print('Gold roi colour embeddings saved to ./cropped/'+config['gold_embeds'])

    # Load already croppped test image rois for compression
    gen_rois = np.load('./cropped/devroisCAT00.npy')

    if args.example:
        plot_example(kmeans_model, gen_rois[random.randint(0, len(gen_rois)-1)])

    else:
        print('\nApplying colour compression')
        c_embeds = apply_colour_compression(kmeans_model, gen_rois)

        # ? save to numpy
        if config['dev_embeds']:
            np.save('./cropped/'+config['dev_embeds'], c_embeds)
            print('Colour embeddings saved to ./cropped/'+config['dev_embeds'])

    print('\n--- done ---')

