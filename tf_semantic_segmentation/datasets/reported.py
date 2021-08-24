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

    def __init__(self, cache_dir):
        super(ReportedDS, self).__init__(cache_dir)
        self._labels = self.labels
        self._colormap = self.colormap

    def raw(self):
        dataset_dir = os.path.join(self.cache_dir)
        if not os.path.exists(dataset_dir):
            extracted = download_records('reported', dataset_dir)
        else:
            extracted = os.path.join(dataset_dir,'dataset')
        imgs = get_files(extracted, extensions=["png","jpg"])
        images = list(filter(lambda x: not x.endswith(".png"), imgs))
        labels = list(filter(lambda x: x.endswith(".png"), imgs))
        trainset = list(zip(images, labels))
        t = get_split(trainset,train_split=1.0)
        v = get_split(trainset,train_split=.5,shuffle=False)
        t = get_split(trainset,train_split=.3,shuffle=False)
        
        return {
            DataType.TRAIN: t[DataType.TRAIN],
            DataType.VAL: t[DataType.TRAIN],
            DataType.TEST: t[DataType.TRAIN]
        }

    @property
    def colormap(self):
        dataset_dir = self.cache_dir
        if not os.path.exists(dataset_dir):
            extracted = download_records('reported', dataset_dir)
        file_path = os.path.join(self.cache_dir, 'labels.txt')
        print("label file path", file_path)
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
        dataset_dir = self.cache_dir
        if not os.path.exists(dataset_dir):
            extracted = download_records('reported', dataset_dir)
        file_path = os.path.join(self.cache_dir, 'labels.txt')

        labels = []

        with open(file_path, "r") as handler:
            for line in handler.readlines():
                args = line.split("\t")
                label = args[-1].replace("\n", "")
                labels.append(label)

        return labels

    def parse_example(self, example):
        image_path, target_path = example
        print("parse",image_path)
        i = imageio.imread(image_path)
        print("parse",target_path)
        t = imageio.imread(target_path)
        mask = np.zeros((i.shape[0], i.shape[1]), np.uint8)        

        for color, label in self._colormap.items():
            print("color",color,"label",label)
            color = [color.r, color.g, color.b]
            idxs = np.where(np.all(t == color, axis=-1))
            mask[idxs] = self._labels.index(label)

        return i, mask


if __name__ == "__main__":
    from .utils import test_dataset

    ds = ReportedDS('/hdd/datasets/reported')
    test_dataset(ds)
