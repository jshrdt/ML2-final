from sklearn.cluster import KMeans
import numpy as np

def get_cluster(ROI_list, n=2):
    # oct 21, 16:32
    #https://github.com/beleidy/unsupervised-image-clustering/blob/master/capstone.ipynb
    ROI_matrix = np.array([np.uint8(img_arr*255).flatten() for img_arr in ROI_list],
                          dtype=np.float32)
    # normalise values
    ROI_matrix /= 255
    #flat_ROIs = KMeans.fit_transform(flat_ROIs) ??

    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(ROI_matrix) #2mins for n=5, 50sec for n=2

    clusters_dict = {cluster_id : list() for cluster_id in set(Y)}
    for i, cluster_id in enumerate(Y):
        clusters_dict[cluster_id].append(ROI_list[i])

    return Y, clusters_dict