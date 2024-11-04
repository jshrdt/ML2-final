# ML2-final

### File descriptions:

C4 preprocess images: 
- call get_cropped_ROIs(data: DataLoader)
-> Reads, resizes images
-> get bounding box using YoloV3, prunes images with nr_of_cats < 1 | nr_of_cats ≥ 2
Returns dict with image_ROIs, box_fail, multi_obj (each list of np.ndarrays)