# loader
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

class DataLoader:
    def __init__(self, data_dir: str, split_data: bool = False):
        self.data_dir = data_dir
        self.imgfiles = self._fetch_files(data_dir)
        self.avg_size = (321, 275)#self._get_avg_size()

        if split_data:
            self.train, self.test = self._split()

    def _fetch_files(self, data_dir):
        print('Fetching files...')
        catfiles = list()
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for fname in sorted(files):
                if fname.endswith('jpg'):
                    catfiles.append(os.path.join(root, fname))
                #elif fname.endswith('cat'):
                 #   pointfiles.append(os.path.join(root, fname))

        return catfiles

    def _split(self):
        #data_df = pd.DataFrame({'imgfiles': self.imgs, 'points': self.points})
        
        # for i, row in data_df.iterrows():
        #     if row['imgfiles'].split('/') in self.gold_file_refs:
        #         non_train_df
        #     raise ValueError

        #train_df = data_df.loc[data_df['imgfiles'] in self.gold_file_refs]

        # Set cutoff points.
        train_len = int(len(self.imgfiles)*0.8)
        #test_len = int(train_len + len(self.gen_imgs)*0.2)

        # Split data.
        train_data = self.imgfiles[:train_len]
        #dev_data = data_df[train_len:dev_len]
        test_data = self.imgfiles[train_len:]

        return train_data, test_data

    def __len__(self):
        return len(self.imgfiles)

    def _get_avg_size(self) -> tuple[int, int]:
        """Find average size of training files to resize input images to."""
        # Collect file widths and heights.
        sizes_w, sizes_h = list(), list()
        for file in self.imgfiles:
            with Image.open(file) as img:
                size = img.size
                sizes_w.append(size[0])
                sizes_h.append(size[1])
        # Get average file size in training dataset.
        avg_size = (round(sum(sizes_w) / len(self.imgfiles)),
                    round(sum(sizes_h) / len(self.imgfiles)))

        return avg_size

def concat_imgs(img_arrays: list) -> np.ndarray:
    """Concat list of image arrays to matrix to facilitate fitting/plots."""
    step = int(len(img_arrays)**0.5+len(img_arrays)**0.2)
    img_matrix = np.concatenate([np.concatenate(img_arrays[i-step:i], axis=0)
                                 for i in range(step, len(img_arrays), step)],
                                axis=1)
    return img_matrix


if __name__=='__main__':
    pass
    #data = DataLoader("./cats/", split_data=True)
