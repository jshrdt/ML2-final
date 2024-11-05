# ML2-final

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

2) fit KMeans colour compression