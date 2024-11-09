from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from C4_helper import concat_imgs

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
parser.add_argument('-lim', '--limit_to', help='limit number of images used',
                    default=False)
parser.add_argument('-clst', '--n_clusters', help='set number of clusters',
                    default=2)

args, unk = parser.parse_known_args()
config = json.load(open(args.config))


LIMIT = int(args.limit_to) if args.limit_to != "False" else False
N_CLUSTERS = int(args.n_clusters)


###  fit model ##
def fit_clusters(embeddings, n=N_CLUSTERS):
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=0)
    kmeans.fit(embeddings) #2mins for n=5, 50sec for n=2

    # save model?

    return kmeans


## apply ####

def get_clusters(model, colour_embeddings):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb

    Y = model.predict(colour_embeddings)
    print([('cluster nr: '+ str(id), 'items in cluster: '+ str(list(Y).count(id)))
           for id in set(Y)])

    return Y

def vis_clusters(cluster_pred, imgs, n_items=21):
    #items = np.array(cluster).reshape(len(cluster), 64, 3)
    #print(cluster.shape)
    # sort rois based on cluster nr
    # include this somewhere else? problem: needs og igms
    clustered_imgs =  {id: [] for id in set(cluster_pred)}
    for i, id in enumerate(Y):
        clustered_imgs[id].append(gold_rois[i])
    
    # simply join to one big img, then do n subplots, one per cluster
    
    fig, ax = plt.subplots(1, N_CLUSTERS)
    fig.set_figwidth(15)
    fig.tight_layout

    for i, imgs in clustered_imgs.items():
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
    # tansfer colour compression into here to get access to all verisons of files
    # better plotting options
    # also means less files to chain and run

    gold_rois = np.load(config["gold_ROIs"])
    gold_embeds = np.load(config['gold_embeds'])
    model = fit_clusters(gold_embeds)

    #gen_rois = np.load(config['devrois'])[:LIMIT]
    #gen_embeds =  np.load(config['dev_embeds'])[:LIMIT]

    Y = get_clusters(model, gold_embeds[:LIMIT])

    vis_clusters(Y, gold_rois)

   # figs = list() #???
    # how to plot n clusters at same time interactively
    #for i in range(N_CLUSTERS-1):
     #   figure, axis = plt.subplots(int(len(clustered_imgs[i])/3)+1, 3)
      #  ax = vis_cluster(clustered_imgs[i], figure)

    # maybe create/drop npys here? want to plot og images for inspection, but info 
    # is lost by now; add some kind of id? use order preserved in files?
    # laod cropped rois again?