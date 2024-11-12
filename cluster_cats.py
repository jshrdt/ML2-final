import argparse
import json
import os

import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from C4_helper import concat_imgs, save_kmeans_model, get_rois
from colour_compression import get_colour_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-lim', '--limit', help='limit number of images used',
                    default=False)
parser.add_argument('-clst', '--n_clusters', help='set number of clusters',
                    default=2)
parser.add_argument('-refit', '--refit_model', help='force new training+saving of model',
                    action='store_true', default=False)
parser.add_argument('-vis', '--visualise', help='plot emerging clusters',
                    action='store_true', default=False)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))

LIMIT = int(args.limit) if args.limit != "False" else False
N_CLUSTERS = int(args.n_clusters)


def get_cluster_model(gold: dict, cluster_nr: int = 2, modelfile: str = None) \
                    -> KMeans:
    """Return KMeans model for colour clustering, either loaded from file
    or newly fit (optional: save to file).

    Args:
        gold (dict): Config dir for gold data.
        cluster_nr (int, optional): Number of clusters to fit model to.
            Defaults to 2.
        modelfile (str, optional): Filename to save model to. Defaults
            to None.

    Returns:
        KMeans: KMeans model for colour clustering
    """
    # Load model for colour clustering...
    if modelfile and os.path.isfile(modelfile) and args.refit_model==False:
        model = pickle.load(open(modelfile, "rb"))
    # or fit a new one.
    else:
        # Get gold embeddings.
        gold_embeddings = get_embeds(gold, gold, limit=False, save=True)
        # Fit new model.
        print('Fitting new clustering model...')
        model = KMeans(n_clusters=cluster_nr, init='k-means++', random_state=0)
        model.fit(gold_embeddings)

        # Save model.
        if (modelfile and os.path.isfile(modelfile)==False) or args.refit_model:
            save_kmeans_model(model, modelfile)

    return model


def get_embeds(test_dir: dict, gold_dir: dict = False, limit: int = False,
               save: bool = False) -> list[np.ndarray]:
    """Create and return colour embeddings from test dir.

    Args:
        test_dir (dict): Config dict for test data.
        gold_dir (dict, optional): Config dict for gold data, not accessed
            when model is loaded from file. Defaults to False.
        limit (int, optional): Limit number of test items. Defaults to 
            False.
        save (bool, optional): Set whether to save embeddigns to file.
            Defaults to False.

    Raises:
        FileNotFoundError: Worst case no reference file.

    Returns:
        list[np.ndarray]: Colour embedding for each input item.
    """
    # Get embeddings...
    if os.path.isfile(test_dir['embeds']):
        # ...a) from file
        print('Loading embeddings from file...', test_dir['embeds'])
        embeddings = np.load(test_dir['embeds'])
        if limit: embeddings = embeddings[:limit]

    # Should always evaluate to True:
    elif os.path.isfile(test_dir['rois']) or os.path.isfile(test_dir['file_refs']):
        # ...b) from transforming gold rois (from file or run preprocessing
        # on files as indiacated by file refs).
        # Defaults to saving ROIs (longest processing time), & embeddings.
        embeddings = get_colour_embeddings(test_dir, gold_dir,
                                           modelfile=gold_dir['compressor_modelfile'],
                                           limit=limit, save=save)
    else:
        raise FileNotFoundError(test_dir['rois'], test_dir['file_refs'],
                                'Config error, neither ROI nor reference files found.')

    return embeddings


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


def cluster_data(gold_dir: dict, test_dir: dict, limit: bool = False,
                 cluster_nr: int = 2, modelfile: str = None,
                 visualise: bool = False):
    """High-level function to cluster data in test dir according to gold dir.

    Args:
        gold_dir (dict): Config dir for gold data.
        test_dir (dict): Config dir for test data.
        limit (int, optional): Limit number of test items. Defaults to
            False.
        cluster_nr (int, optional): Number of clusters. Defaults to 2.
        modelfile (str, optional): File for cluster model. Defaults to None.
        visualise (bool, optional): Set whether to plot test data clusters.
            Defaults to False.

    Returns:
        _type_: _description_
    """
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb

    # Get model for colour-based clustering
    model = get_cluster_model(gold_dir, cluster_nr, modelfile)
    # Get test embeddings
    test_embeddings = get_embeds(test_dir, gold_dir, limit=limit, save=True)

    # Check if dimensionalities of test embeds and model align
    if len(test_embeddings[0]) != model.n_features_in_:
        dimensionality_mismatch(test_embeddings, model)

    # Cluster embeddings and get predicted cluster id per of embedding.
    Y = model.predict(test_embeddings)

    if visualise:
        # Plot test data
        rois = get_rois(test_dir, limit=limit)
        visualise_clusters(Y, rois)

    return Y


if __name__=='__main__':
    gold_dir = config['CAT_00_solid']
    test_dir = config['CAT_01']
    modelfile = gold_dir['cluster_modelfile']

    # Cluster data
    Y = cluster_data(gold_dir, test_dir, limit=LIMIT, cluster_nr=N_CLUSTERS,
                    modelfile=modelfile, visualise=args.visualise)

    print('\n--- done ---')
