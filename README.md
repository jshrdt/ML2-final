# ML2-final

Dataset: https://www.kaggle.com/datasets/crawford/cat-dataset/data, Oct 9 2024, 11:05am

"super small projects"

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