from .dataset import Dataset
from ..utils import get_files, download_and_extract, download_file
from .utils import get_split, DataType, Color, download_records

import imageio
import os
import numpy as np


class ReportedDS(Dataset):
    """
    Image Segmentation DataSet of Road Scenes

    Dataset url: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip
    """

    DATA_URL = "https://drive.google.com/file/d/1bJ7HedSEsmeCWb-AyZ7WNndfJ6OEI3Vu/view?usp=sharing"
    LABEL_COLORS_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt"

    def __init__(self, cache_dir):
        super(ReportedDS, self).__init__(cache_dir)
        self._labels = self.labels
        self._colormap = self.colormap

    def raw(self):
        dataset_dir = os.path.join(self.cache_dir, 'dataset')
        extracted = download_records('reported', dataset_dir)
        imgs = get_files(extracted, extensions=["png","jpg"])
        images = list(filter(lambda x: not x.endswith(".png"), imgs))
        labels = list(filter(lambda x: x.endswith(".png"), imgs))
        trainset = list(zip(images, labels))
        return get_split(trainset)

    @property
    def colormap(self):
        file_path = os.path.join(self.cache_dir, 'dataset/reported/label_colors.txt')

        color_label_mapping = {}
        with open(file_path, "r") as handler:
            for line in handler.readlines():
                args = line.split("\t")
                color = list(map(lambda x: int(x), args[0].split(" ")))
                color = Color(*color)
                label = args[-1].replace("\n", "")
                color_label_mapping[color] = label

        return color_label_mapping

    @property
    def labels(self):
        file_path = os.path.join(self.cache_dir, 'dataset/reported/label_colors.txt')

        labels = []

        with open(file_path, "r") as handler:
            for line in handler.readlines():
                args = line.split("\t")
                label = args[-1].replace("\n", "")
                labels.append(label)

        return labels

    def parse_example(self, example):
        image_path, target_path = example
        i = imageio.imread(image_path)
        t = imageio.imread(target_path)
        mask = np.zeros((i.shape[0], i.shape[1]), np.uint8)

        for color, label in self._colormap.items():
            color = [color.r, color.g, color.b]
            idxs = np.where(np.all(t == color, axis=-1))
            mask[idxs] = self._labels.index(label)

        return i, mask


if __name__ == "__main__":
    from .utils import test_dataset

    ds = ReportedDS('/hdd/datasets/reported')
    test_dataset(ds)
