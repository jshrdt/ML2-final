# ML2-final

Dataset: https://www.kaggle.com/datasets/crawford/cat-dataset/data, Oct 9 2024, 11:05am

"super small projects"

___

Minimum file requirements: CAT00 folder, file refs

Recommendation for testing:  

The fastest way to run the files & view their main functionality; necessary files (ROI arrays/embeddings) will be created & saved on execution. As preprocessing of images to ROI arrays takes the longest, pass -lim (int) to limit the amount of test items to be processed (only relevant to reduce time if 'rois' file  does not already exists).

(Optional but recommended: $ python3 preprocess_imgs.py & python3 preprocess_imgs.py -lim 100 -srcdir ./cats/CAT_01)


$ python colour_compression.py -ex -lim 10, runtime ca 4secs from scratch ##8to create rows for gold embeddings
-> plots random img's ROI, ROI post colour compression, its colour profile and the full compression palette

$ python cluster_cats.py -vis -lim 80 -clst 4, runtime ca 8min from scratch, 4mins if colour_compression was run previously (gold embeds) saved
-> plots clusters (4) for 80 imgs from test dir




####
Run preprocess: imgdir -> ROI nps

1 train col compressor & save model
2 col compress ROIs (?get col profiles)

1 Train kmeans colour clustering (input col profiles, handles vectorisation)
2 apply clustering 

Question of. Eval

? Save profiles as json

? Write main exec script that reads gold+gen np data and then fit+compress+fit+clusters?

### File descriptions:

C4 preprocess images: 
- call get_cropped_ROIs(data: DataLoader)
-> Reads, resizes images
-> get bounding box using YoloV3, prunes images with nr_of_cats < 1 | nr_of_cats ≥ 2
Returns dict with image_ROIs, box_fail, multi_obj (each list of np.ndarrays)


gold_cropped_imgs: list of hand selected files with good grabCut results used in fitting KMeans for colour compression

__
Intended flow:

1) Transform CAT00: cut gold images & gen images
Ok to load as normal tbh
! Can't save together with numpy file
-> preprocess (get img_dict), then filter into golds/general and save both files
Filenames are preserved through preprocessing, 

# Object detection
Goal for object detection with bboxes  
Simplyfying assumption:  
No other animals than cats, 1 animal at a time  

1. Resize to half of average size in dir, to increase processing speed
2. Get bbox (if only 1 box, assume label cat), for 2+ labels, only keep if exactly one cat label is present
3. Remove background with grabCut
4. Compress colours using K-Means (k=64)



2) fit KMeans colour compression