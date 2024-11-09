from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pickle
import os

from C4_helper import concat_imgs, save_kmeans_model
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

MODELFILE =  config['cluster_modelfile']
LIMIT = int(args.limit) if args.limit != "False" else False
N_CLUSTERS = int(args.n_clusters)


def get_cluster_model(cluster_nr=2, modelfile=None):
    if modelfile and os.path.isfile(modelfile) and args.refit_model==False:
        # Load model for colour clustering...
        model = pickle.load(open(modelfile, "rb"))
    else:
        # or fit a new model.
        print('Fitting new model...')
    # create from scratch to avoid discrepancies
        gold_rois = np.load(config['gold_ROIs'])
        gold_embeddings = get_colour_embeddings(gold_rois)

        model = KMeans(n_clusters=cluster_nr, init='k-means++', random_state=0)
        model.fit(gold_embeddings) #2mins for n=5, 50sec for n=2

        if (modelfile and os.path.isfile(modelfile)==False) or args.refit_model:
            save_kmeans_model(model, modelfile)


    return model

## apply ####

def cluster_data(test_rois, cluster_nr=2, modelfile=None):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb

    model = get_cluster_model(cluster_nr, modelfile)


    test_data_embeds = get_colour_embeddings(test_rois)

    # # ??
    # if type(test_arrays_file)==str:
    #     test_arrays = np.load(test_arrays_file) # can also be created in-place

    Y = model.predict(test_data_embeds)


    print([('cluster nr: '+ str(id), 'items in cluster: '+ str(list(Y).count(id)))
           for id in set(Y)])


    return Y


def visualise_clusters(Y, imgs):
    # Sort images by cluster
    clustered_imgs =  {id: [] for id in set(Y)}
    for i, id in enumerate(Y):
        clustered_imgs[id].append(imgs[i])

    # Create plots
    fig = plt.figure()
    gs = fig.add_gridspec(1, len(set(Y)))

    # plot clusters
    for i, imgs in clustered_imgs.items():
        joined_imgs = concat_imgs(imgs)
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(joined_imgs)

    plt.show()

if __name__=='__main__':
    ## RESTRUCTURE:
   # Q? how to deal with grabcut fails -> col compression fails

    if os.path.isfile(config['gold_ROIs']):
        # Load pre-processed (cropped) test image rois.
        test_rois = np.load(config['gold_ROIs'])
    else:
        # try to create?; same as in col compression
        raise FileNotFoundError('No image ROIs found at', config['gold_ROIs'])

    ## tbd shuffle?
    if LIMIT: test_rois = test_rois[:LIMIT]

    # change on: load gen/gold_rois here, transform within function
    Y = cluster_data(test_rois, cluster_nr=N_CLUSTERS,
                    modelfile=MODELFILE)
    if args.visualise:
        visualise_clusters(Y, test_rois)

    # maybe create/drop npys here? want to plot og images for inspection, but info 
    # is lost by now; add some kind of id? use order preserved in files?
    # laod cropped rois again?
    #print('\n--- done ---')
