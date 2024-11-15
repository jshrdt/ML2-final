# ML2-final

___
### Package requirements:  
argparse, json, os, random, warnings, collections, matplotlib.pyplot, numpy, pandas, pickle, sklearn, tqdm

Only for preprocessing (not strictly required, preprocessed files are supplied in repository)
Modules cv2, cvlib, PIL, torch
External files: CAT_00 and CAT_01 folders from https://www.kaggle.com/datasets/crawford/cat-dataset/data

___

### File breakdown:

config.json
* Specifies paths for loading/saving

helper.py  
* some loading/processing functions shared across files
* no main function

process_imgs.py  
* Loading images from files
* Object detection
* GrabCut image segmentation
* Saving ROI arrays for input data dir to .npy

colour_compression.py  
* Load ROI arrays
* Fit+save KMeans model & apply colour compression 
* Extract colour profiles from compressed ROI arrays
* Plot example for output of colour compression & visualisation of colour profile

cluster_cats.py  
* Load ROI arrays, perform feature extraction according to colour compression
* Vectorise colour profiles
* Fit+save KMeans model & apply clustering
* Plots clusters of test data

____

### Recommendation for testing, files no particular order:  

First: Unzip .npy files

> $ python feature_extraction.py
* Fits and saves a KMeans (k=32) colour compression model on compressed ROIs from CAT_00_solid
* picks one random ROI array from the CAT_01_rois.npy file created in (2) and runs the feature extraction steps; plots (1.1) the original ROI array, (1.2) the colour compressed ROI array, (1.3) the associated full colour palette the colour compression model extracted from the gold set CAT_00_solid and used in compression, and (2) the colour profile of the exemplary compressed ROI array (a relative frequency distribution of colour centroids, min freq > 0.01%).


> § python cluster_cats.py -lim 100 -clst 4
* Fits and saves a KMeans (k=4) clustering model on colour embeddings from CAT_00_solid
* Clusters and plots the first 100 image ROIs from CAT_01


To change which data is used for model fitting and which are used as test data in feature_extraction.py and cluster_cats.py, set the gold_dir and test_dir variables in the main program to any of these: config['CAT_00_solid'], config['CAT_00_mixed'], config['CAT_01'].

Command line arguments are as follows; any arguments specified for the feature extraction script can be internally passed along when executing the cluster_cats script

* feature_extraction.py  
--train_new (-new): Fit+save a new colour compression model (default=False)  
--n_colours (-col): Set max number of colours across dataset to compress images to (default=32)  

* cluster_cats.py  
--refit_model (-refit): Fit+save a new clustering model (default=False)  
--n_clusters (-clst): Set number of clusters (default=2)  
--limit (lim): Set max number of ROI arrays from test data to predict/plot (default=False)  

___

Since preprocessing takes some time & relies on cv/cvlib which uses tensor flow, I have decided to supply toy samples of the ROI files for easy testing of the code. Preprocessing was done using these commands and requires some more modules/files as detailed above. To run the code yourself, uncomment the 'from ...import...' statement and second half of the get_rois() function in helper.py. I recommend passing -lim X to reduce the nr of items processed from CAT_01 in particular.

> $ python process_imgs.py  
* handles extraction of ROI arrays for gold sets CAT_00_solid and CAT_00_mixed, saves these as .npy files to roi path in config

> $ python process_imgs.py -srcdir ./cats/CAT_01
* handles extraction of ROI arrays for the first 200 images in blind test set CAT_01, saves these as .npy files to roi path in config
