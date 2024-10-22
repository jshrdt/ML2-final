# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

def compress_colours(img_array, n_colors=64):

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow works well on float data (need to
    # be in the range [0-1])
    img_array = np.array(img_array, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img_array.shape)
    assert d == 3
    image_array = np.reshape(img_array, (w * h, d))

    # print("Fitting model on a small sub-sample of the data")
    #t0 = time()
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    #print(f"done in {time() - t0:0.3f}s.")

    # Get labels for all points
    #print("Predicting color indices on the full image (k-means)")
    #t0 = time()
    labels = kmeans.predict(image_array)
    #print(f"done in {time() - t0:0.3f}s.")

    compressed_img = recreate_image(kmeans.cluster_centers_, labels, w, h)

    # codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
    # print("Predicting color indices on the full image (random)")
    # t0 = time()
    # labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
    # print(f"done in {time() - t0:0.3f}s.")


    # Display all results, alongside original image
    # plt.figure(1)
    # plt.clf()
    # plt.axis("off")
    # plt.title("Original image (96,615 colors)")
    # plt.imshow(img_array)

    # plt.figure(2)
    # plt.clf()
    # plt.axis("off")
    # plt.title(f"Quantized image ({n_colors} colors, K-Means)")
    # plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

    # plt.figure(3)
    # plt.clf()
    # plt.axis("off")
    # plt.title(f"Quantized image ({n_colors} colors, Random)")
    # plt.imshow(recreate_image(codebook_random, labels_random, w, h))
    # plt.show()

    return compressed_img

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1) 