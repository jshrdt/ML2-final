import argparse
import json
import pickle
import os
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
parser.add_argument('-col', '--n_colours', help='compress and plot random example',
                    default=33)


args, unk = parser.parse_known_args()
config = json.load(open(args.config))

LIMIT = int(args.limit) if args.limit != "False" else False
N_CLUSTERS = int(args.n_clusters)

def get_cluster_model(gold, cluster_nr: int = 2, modelfile: str = None) -> KMeans:
    """Return KMeans model for colour clustering, either loaded from file
    or newly fit (optional: save to file).

    Args:
        cluster_nr (int, optional): Number of clusters to fit model to.
            Defaults to 2.
        modelfile (str, optional): Filename to save model to. Defaults
            to None.

    Returns:
        KMeans: KMeans model for colour clustering.
    """
    # Load model for colour clustering...
    if modelfile and os.path.isfile(modelfile) and args.refit_model==False:
        model = pickle.load(open(modelfile, "rb"))
    # or fit a new one.
    else:
        print('Preparing...')
        # Get gold embeddings...
        gold_embeddings = get_embeds(gold, gold, limit=False, save=True)

        print('Fitting new clustering model...')
        model = KMeans(n_clusters=cluster_nr, init='k-means++', random_state=0)
        model.fit(gold_embeddings) #2mins for n=5, 50sec for n=2

        if (modelfile and os.path.isfile(modelfile)==False) or args.refit_model:
            save_kmeans_model(model, modelfile)

    return model

## apply ####

def get_embeds(test_dir, gold_dir=False, limit=False, save=False):
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
                                           limit=limit, save=save) #TBD
    else:
        raise FileNotFoundError(test_dir['rois'], test_dir['file_refs'],
                                'Config error, neither ROI nor reference files found.')
    return embeddings


def dimensionality_mismatch(test_embeds, model):
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


def cluster_data(gold_dir, test_dir, limit=False, cluster_nr=2, modelfile=None,
                 visualise=False):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb

    # Get model for colour-based clustering
    model = get_cluster_model(gold_dir, cluster_nr, modelfile)
    # Get test embeddings
    test_embeddings = get_embeds(test_dir, gold_dir, limit=limit, save=True)

    # check if dimensionalities align
    if len(test_embeddings[0]) != model.n_features_in_:
        dimensionality_mismatch(test_embeddings, model)

    # Cluster embeddings and get predicted cluster id per of embedding.
    Y = model.predict(test_embeddings)

    if visualise:
        rois = get_rois(test_dir, limit=limit)

        visualise_clusters(Y, rois)

    return Y


def visualise_clusters(Y, imgs):
    # Sort images by cluster
    
    clustered_imgs =  {id: [] for id in set(Y)}
    for i, id in enumerate(Y):
        clustered_imgs[id].append(imgs[i])

    # Create plots
    fig = plt.figure()
    fig.suptitle(f'KMeans colour clusters, ({len(imgs)} samples)')
    gs = fig.add_gridspec(1, len(set(Y)))

    # plot clusters
    for i, imgs in clustered_imgs.items():
        joined_imgs = concat_imgs(imgs)
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(joined_imgs)
        ax.set_title(f'n={len(imgs)}')

    plt.show()

if __name__=='__main__':
    gold_dir = config['CAT_00_solid']
    test_dir = config['CAT_01']
    modelfile = gold_dir['cluster_modelfile']

    # Cluster data
    Y = cluster_data(gold_dir, test_dir, limit=LIMIT, cluster_nr=N_CLUSTERS,
                    modelfile=modelfile, visualise=args.visualise)


    # maybe create/drop npys here? want to plot og images for inspection, but info 
    # is lost by now; add some kind of id? use order preserved in files?
    # laod cropped rois again?
    print('\n--- done ---')
