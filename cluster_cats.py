from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# def get_cluster(ROI_list, n=2):
#     # oct 21, 16:32
#     #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb
#     ROI_matrix = np.array([np.uint8(img_arr*255).flatten() for img_arr in ROI_list],
#                           dtype=np.float32)
#     # normalise values
#     ROI_matrix /= 255
#     #flat_ROIs = KMeans.fit_transform(flat_ROIs) ??

#     kmeans = KMeans(n_clusters=n, init='k-means++', random_state=0)
#     Y = kmeans.fit_predict(ROI_matrix) #2mins for n=5, 50sec for n=2

#     clusters_dict = {cluster_id : list() for cluster_id in set(Y)}
#     for i, cluster_id in enumerate(Y):
#         clusters_dict[cluster_id].append(ROI_list[i])

#     return Y, clusters_dict



# TBD train cluster KMEANS vs test ##

## TBD edit ####

def get_ccluster(colour_embeddings, n=2):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb
   
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(colour_embeddings) #2mins for n=5, 50sec for n=2

    return Y

def vis_cluster(cluster, n_items=21):
    #items = np.array(cluster).reshape(len(cluster), 64, 3)
    #print(cluster.shape)
    clen = len(cluster)
    figure, axis = plt.subplots(int(clen/3)+1, 3)
    figure.set_figwidth(8)
    figure.set_figheight(15)

    for i, img in enumerate(cluster):
        if i < int(clen/3):
            axis[i][0].imshow(img, interpolation='nearest')
        elif i < int(clen/3*2):
            axis[i-int(clen/3)][1].imshow(img, interpolation='nearest')
        else: 
            axis[i-int(clen/3*2)][2].imshow(img, interpolation='nearest')

    plt.show()
    
if __name__=='__main__':
    #! throw out (0,0,0) colour
    pass