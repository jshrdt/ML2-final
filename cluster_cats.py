import argparse
import json
import os

import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

from helper import concat_imgs, save_kmeans_model, get_rois
from feature_extraction import get_compress_model, compress_roi, get_colour_profile


parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-lim', '--limit', help='limit number of images used',
                    default=False)
parser.add_argument('-clst', '--n_clusters', help='set number of clusters',
                    default=2)
parser.add_argument('-refit', '--refit_model', help='force new training+saving of model',
                    action='store_true', default=False)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

LIMIT = int(args.limit) if args.limit != "False" else False
N_CLUSTERS = int(args.n_clusters)


def get_cluster_model(gold: dict, cluster_nr: int = 2)  -> KMeans:
    """Return KMeans model for colour clustering, either loaded from file
    or newly fit (optional: save to file).

    Args:
        gold (dict): Config dir for gold data.
        cluster_nr (int, optional): Number of clusters to fit model to.
            Defaults to 2.

    Returns:
        KMeans: KMeans model for colour clustering
    """
    # Load model for colour clustering...
    if os.path.isfile(gold['cluster_modelfile']) and not args.refit_model:
        model = pickle.load(open(gold['cluster_modelfile'], "rb"))
    # or fit a new one.
    else:
        # Get gold embeddings.
        gold_embeds_raw = get_embeds(gold, gold)['colour_embeddings']
        gold_embeddings = [embed for embed in gold_embeds_raw if sum(embed)>0]
        # Fit new model.
        print('Fitting new clustering model on embeds from', gold['rois'])
        model = KMeans(n_clusters=cluster_nr, init='k-means++', random_state=0)
        model.fit(gold_embeddings)

        # Save model.
        if not os.path.isfile(gold['cluster_modelfile']) or args.refit_model:
            save_kmeans_model(model, modelfile)

    return model


def get_embeds(test_dir: dict, gold_dir: dict = False,
               limit: int = False) -> list[np.ndarray]:
    """Create and return colour embeddings from test dir.

    Args:
        test_dir (dict): Config dict for test data.
        gold_dir (dict, optional): Config dict for gold data, not accessed
            when model is loaded from file. Defaults to False.
        limit (int, optional): Limit number of test items. Defaults to 
            False.

    Raises:
        FileNotFoundError: Worst case no reference file.

    Returns:
        list[np.ndarray]: Colour embedding for each input item.
    """
    # Initialise container for organising data.
    data = {'rois': list(), 'compressed_rois': list(), 'colour_profiles': list(),
           'colour_embeddings': list()}

    # Get ROIs, get model for colour compression and extract colour palette.
    test_rois = get_rois(test_dir, limit)
    compression_model = get_compress_model(gold_dir)
    palette = [tuple(np.uint8(col))
               for col in compression_model.cluster_centers_]

    print('Transforming data...')
    # Iterative over ROI arrays, perform colour compression and vectorisation,
    # store and initial, medial, and final representation in parallel dict.
    for roi in tqdm(test_rois):
        # Colour compression
        compressed_roi = compress_roi(roi, compression_model)
        # Colour profile creation
        profile = get_colour_profile(compressed_roi, palette)
        # Vectorisation
        embedding = np.array(list(profile.values()))
        # Store data
        if sum(embedding)>0:
            data['rois'].append(roi)
            data['compressed_rois'].append(compressed_roi)
            data['colour_profiles'].append(profile)
            data['colour_embeddings'].append(embedding)

    return data


def dimensionality_mismatch(test_embeds, model):
    # Detailed error why script might fail across runs.
    print('-'*80,
          f'Error: Test embeddings input has {len(test_embeds[0])} features,'
          +f' but KMeans was fitted for {model.n_features_in_} features',
          '-'*80,
          'Solutions:',
          f'(1) Create new test embeddings with n={model.n_features_in_} features ($ python colour_compression.py -new -s -col {model.n_features_in_})',
          f'(2) Create new gold embeddings with n={len(test_embeds[0])} features & refit cluster model',
          '-'*80,
          sep='\n')
    raise ValueError('Dimensionality mismatch')


def visualise_clusters(Y, imgs):
    """Plot original ROIs in clusters predicted with colour embeddings."""
    # Sort images by cluster
    clustered_imgs =  {id: [] for id in set(Y)}
    for i, id in enumerate(Y):
        clustered_imgs[id].append(imgs[i])

    # Create plots
    fig = plt.figure()
    fig.suptitle(f'KMeans colour clusters, ({len(imgs)} samples)')
    gs = fig.add_gridspec(1, len(set(Y)))

    # Plot clusters
    for i, imgs in clustered_imgs.items():
        joined_imgs = concat_imgs(imgs)
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(joined_imgs)
        ax.set_title(f'n={len(imgs)}')

    plt.show()


def cluster_data(test_dir: dict, gold_dir: dict, limit: bool = False,
                 cluster_nr: int = 2) -> np.ndarray:
    """High-level function to cluster data in test dir according to gold dir.

    Args:
        test_dir (dict): Config dir for test data.
        gold_dir (dict): Config dir for gold data.
        limit (int, optional): Limit number of test items. Defaults to
            False.
        cluster_nr (int, optional): Number of clusters. Defaults to 2.

    Returns:
        np.ndarray: Array of cluster predictions for test data.
    """
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb

    # Get model for colour-based clustering
    model = get_cluster_model(gold_dir, cluster_nr)

    # Get test embeddings
    data_dict = get_embeds(test_dir, gold_dir, limit=limit)

    # Check if dimensionalities of test embeds and model align, raise Error if not
    if len(data_dict['colour_embeddings'][0]) != model.n_features_in_:
        dimensionality_mismatch(data_dict['colour_embeddings'], model)

    # Cluster embeddings and get predicted cluster id per of embedding.
    Y = model.fit_predict(data_dict['colour_embeddings'])

    return Y, data_dict['rois']


if __name__=='__main__':
    gold_dir = config['CAT_00_solid']
    test_dir = config['CAT_00_solid']#config['CAT_01']
    modelfile = gold_dir['cluster_modelfile']

    # Cluster data
    Y, rois = cluster_data(test_dir, gold_dir, limit=LIMIT,
                           cluster_nr=N_CLUSTERS)

    # Plot data
    visualise_clusters(Y, rois)

    print('\n--- done ---')
