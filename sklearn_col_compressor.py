# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

#from time import time

# import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.cluster import KMeans
#from sklearn.datasets import load_sample_image
# from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning

### edited by moi:

import json
import warnings
warnings.filterwarnings("error", category=ConvergenceWarning)

import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
args, unk = parser.parse_known_args()
config = json.load(open(args.config))

#split into trainign and application
# import pickle

# filename = "my_model.pickle"

# # save model
# pickle.dump(model, open(filename, "wb"))

# # load model
# loaded_model = pickle.load(open(filename, "rb"))

def format_arr(img_array):
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow works well on float data (need to
    # be in the range [0-1])

    arr_norm = np.array(img_array, dtype=np.uint8)
    # Load Image and transform to a 2D numpy array.

    w, h, d = tuple(arr_norm.shape)
    assert d == 3
    arr_norm = np.reshape(arr_norm, (w * h, d))

    return arr_norm, w, h, d


def train_col_compressor(img_matrix, n_colours=65, modelfile=None):
    print('Formatting data...')
    img_matrix_norm, w, h, d = format_arr(img_matrix)
    # img_matrix_norm = np.array([pix for pix in img_matrix_norm
    #                             if sum(pix)>0]) #takes too long

    # print("Fitting model on a small sub-sample of the data")
    #t0 = time()
    try:
        print('Fitting model...')
        image_matrix_sample = shuffle(img_matrix_norm, random_state=0,
                                      n_samples=1_000)
        kmeans_model = KMeans(n_clusters=n_colours,
                              random_state=0).fit(image_matrix_sample)
    except ConvergenceWarning:
        return 0
    
    if modelfile: 
        # save model
        pickle.dump(kmeans_model, open(modelfile, "wb"))
        
    #img_matrix_norm = np.uint8(img_matrix_norm*255)  ## return to normal format

    return kmeans_model

# # load model
# loaded_model = pickle.load(open(modelfile, "rb"))
    ## save/return model
    


def compress_colours(model: KMeans, img_array: np.ndarray):

    image_array, w, h, d = format_arr(img_array)

    # Get labels for all points
    #print("Predicting color indices on the full image (k-means)")
    #t0 = time()
    labels = model.predict(image_array)
    #print(f"done in {time() - t0:0.3f}s.")

    compressed_img = recreate_image(model.cluster_centers_, labels, w, h)
    #compressed_img_range255 = np.uint8(compressed_img*255)

    # codebook_random = shuffle(image_array, random_state=0, n_samples=n_colours)
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
    # plt.title(f"Quantized image ({n_colours} colors, K-Means)")
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

def get_colour_embedding(model: KMeans, img_arr: np.ndarray):
    compressed_roi = np.uint8(compress_colours(model, img_arr))
    gold_palette = np.array([np.uint8(col) for col in model.cluster_centers_
                         if sum(np.uint8(col)>0)])
    c_embedding = {tuple(colour): 0  for colour in list(gold_palette)}
    for x in compressed_roi:
        for colour in x:
            if sum(colour) >0:
                c_embedding[tuple(colour)] = c_embedding.get(tuple(colour), 0)+1
            
    return c_embedding
    

# def compress_colours(img_array, n_colours=64):

#     # Convert to floats instead of the default 8 bits integer coding. Dividing by
#     # 255 is important so that plt.imshow works well on float data (need to
#     # be in the range [0-1])
#     img_array = np.array(img_array, dtype=np.float64) / 255

#     # Load Image and transform to a 2D numpy array.
#     w, h, d = original_shape = tuple(img_array.shape)
#     assert d == 3
#     image_array = np.reshape(img_array, (w * h, d))

#     # print("Fitting model on a small sub-sample of the data")
#     #t0 = time()
#     try:
#         image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
#         kmeans = KMeans(n_clusters=n_colours, random_state=0).fit(image_array_sample)
#     except ConvergenceWarning:
#         return 0
#     #print(f"done in {time() - t0:0.3f}s.")

#     # Get labels for all points
#     #print("Predicting color indices on the full image (k-means)")
#     #t0 = time()
#     labels = kmeans.predict(image_array)
#     #print(f"done in {time() - t0:0.3f}s.")

#     compressed_img = recreate_image(kmeans.cluster_centers_, labels, w, h)
#     compressed_img_range255 = np.uint8(compressed_img*255)

#     # codebook_random = shuffle(image_array, random_state=0, n_samples=n_colours)
#     # print("Predicting color indices on the full image (random)")
#     # t0 = time()
#     # labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
#     # print(f"done in {time() - t0:0.3f}s.")


#     # Display all results, alongside original image
#     # plt.figure(1)
#     # plt.clf()
#     # plt.axis("off")
#     # plt.title("Original image (96,615 colors)")
#     # plt.imshow(img_array)

#     # plt.figure(2)
#     # plt.clf()
#     # plt.axis("off")
#     # plt.title(f"Quantized image ({n_colours} colors, K-Means)")
#     # plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

#     # plt.figure(3)
#     # plt.clf()
#     # plt.axis("off")
#     # plt.title(f"Quantized image ({n_colors} colors, Random)")
#     # plt.imshow(recreate_image(codebook_random, labels_random, w, h))
#     # plt.show()

#     return compressed_img_range255

# def recreate_image(codebook, labels, w, h):
#     """Recreate the (compressed) image from the code book & labels"""
#     return codebook[labels].reshape(w, h, -1)

if __name__=='__main__':
    # Train/for KMeans for colour compression
    # load in gold files
    gold_rois = np.load(config['gold_img_ROIs_file'])
    print(len(gold_rois))