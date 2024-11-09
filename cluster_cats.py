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
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-lim', '--limit_to', help='limit number of images used',
                    default=False)
parser.add_argument('-clst', '--n_clusters', help='set number of clusters',
                    default=2)
parser.add_argument('-new', '--train_new', help='force new training+saving of model',
                    action='store_true', default=False)
parser.add_argument('-vis', '--visualise', help='plot emerging clusters',
                    action='store_true', default=False)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))


LIMIT = int(args.limit_to) if args.limit_to != "False" else False
N_CLUSTERS = int(args.n_clusters)
MODELFILE =  config['cluster_modelfile']


def get_cluster_model(cluster_nr=2, modelfile=None):
    if modelfile and os.path.isfile(modelfile) and args.train_new==False:
        # Load model for colour clustering...
        model = pickle.load(open(modelfile, "rb"))
    else:
        # or fit a new model.
        print('Fitting new model...')
        gold_embeddings = np.load(config["gold_embeds"])
        model = KMeans(n_clusters=cluster_nr, init='k-means++', random_state=0)
        model.fit(gold_embeddings) #2mins for n=5, 50sec for n=2

        if (modelfile and os.path.isfile(modelfile)==False) or args.train_new:
            save_kmeans_model(model, modelfile)


    return model

## apply ####
def transform_data(roi_file, limit):

    test_embeds = get_colour_embeddings(roi_file, limit=limit)



def cluster_data(test_arrays_file, cluster_nr=2, limit=False, modelfile=None,
                 visualise=False):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb

    model = get_cluster_model(cluster_nr, modelfile)

    # NEW FUNC: load in gen/gold rois & transofmr into colour vectors, 
    # keep in pd df, so that when color compression fails, u dont end 
    # up with non-parallel lists (will break plotting)
    
    test_data_embeds = transform_data(test_arrays_file, limit=limit)
    
    
    # # ??
    # if type(test_arrays_file)==str:
    #     test_arrays = np.load(test_arrays_file) # can also be created in-place

    # if LIMIT:
    #     test_arrays = test_arrays[:LIMIT]

    # Y = model.predict(test_arrays)


    # print([('cluster nr: '+ str(id), 'items in cluster: '+ str(list(Y).count(id)))
    #        for id in set(Y)])

    # if visualise:
    #     pass
    #     visualise_clusters(Y, test_arrays)

    return 0



def visualise_clusters(cluster_pred, imgs, n_items=21):
    #items = np.array(cluster).reshape(len(cluster), 64, 3)
    #print(cluster.shape)
    # sort rois based on cluster nr
    # include this somewhere else? problem: needs og igms
    clustered_imgs =  {id: [] for id in set(cluster_pred)}
    for i, id in enumerate(cluster_pred):
        clustered_imgs[id].append(imgs[i])
    
    # simply join to one big img, then do n subplots, one per cluster
    
    fig, ax = plt.subplots(1, N_CLUSTERS)
    fig.set_figwidth(15)
    fig.tight_layout

    for i, imgs in clustered_imgs.items():
       # print(i, N_CLUSTERS)
        joined_imgs = concat_imgs(imgs)
        ax[i].imshow(joined_imgs)
    
    
    # clen = len(cluster)
    
    # axis = fig.subplots(int(clen/3)+1, 3)

    # #figure.set_figwidth(8)
    # #figure.set_figheight(15)
    # #figure.tight_layout

    # for i, img in enumerate(cluster):
    #     if i < int(clen/3):
    #         axis[i][0].imshow(img, interpolation='nearest')
    #     elif i < int(clen/3*2):
    #         axis[i-int(clen/3)][1].imshow(img, interpolation='nearest')
    #     else: 
    #         axis[i-int(clen/3*2)][2].imshow(img, interpolation='nearest')
    plt.show()



if __name__=='__main__':

    ## RESTRUCTURE:
    # as some items are dropped during running:
    # shuffle test data dir, chose up to LIMIT cropped ROIS
    # run vectorisation from here; keep in data structure
    # so that genrois and embeds & Y are aligned
    # need to start with gen 

    # change on: load gen/gold_rois here, transform within function
    Y = cluster_data(config['gold_ROIs'], cluster_nr=N_CLUSTERS,
                     limit=LIMIT, modelfile=MODELFILE, visualise=args.visualise)


    # maybe create/drop npys here? want to plot og images for inspection, but info 
    # is lost by now; add some kind of id? use order preserved in files?
    # laod cropped rois again?
    print('\n--- done ---')
