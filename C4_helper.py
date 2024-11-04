import os
import pandas as pd

from PIL import Image

from tqdm import tqdm


class DataLoader:
    def __init__(self, data_dir: str, split_data: bool = False):
        self.imgs, self.points = self._read_files(data_dir)
        if split_data:
            self.train, self.dev, self.test = self._split()
            self.avg_size = self._get_avg_size(self.train['imgs'])
        else:
            self.avg_size = self._get_avg_size(self.imgs)

    def _read_files(self, data_dir):
        print('Fetching files...')
        catfiles, pointfiles = list(), list()
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for fname in sorted(files):
                if fname.endswith('jpg'):
                    catfiles.append(os.path.join(root, fname))
                elif fname.endswith('cat'):
                    pointfiles.append(os.path.join(root, fname))

        return catfiles, pointfiles

    def _split(self):
        data_df = pd.DataFrame({'imgs': self.imgs, 'points': self.points})

        # Set cutoff points.
        train_len = int(len(data_df)*0.8)
        dev_len = int(train_len + len(data_df)*0.1)

        # Split data.
        train_data = data_df[:train_len]
        dev_data = data_df[train_len:dev_len]
        test_data = data_df[dev_len:]

        return train_data, dev_data, test_data

    def __len__(self):
        return len(self.imgs)

    def _get_avg_size(self, data: list) -> tuple[int, int]:
        """Find average size of training files to resize input images to."""
        # Collect file widths and heights.
        sizes_w, sizes_h = list(), list()
        for file in data:
            with Image.open(file) as img:
                size = img.size
                sizes_w.append(size[0])
                sizes_h.append(size[1])
        # Get average file size in training dataset.
        avg_size = (round(sum(sizes_w) / len(data)),
                    round(sum(sizes_h) / len(data)))

        return avg_size
if __name__=='__main__':
    pass