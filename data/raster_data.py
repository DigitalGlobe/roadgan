import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from pathlib import Path
from random import shuffle, seed, randint, Random, random

from tilelib import TileEngine, OSMRoadEngine
from tqdm import tqdm
from mercantile import children, parent, Tile
import json

tile_engine = TileEngine()
tile_engine.add_directory("data/tilestacks")

road_engine = OSMRoadEngine(use_overpass=False)
road_engine.load_directory("data/cache", verbose=True)


def make_transform():
    r90 = randint(0, 3)
    vflip = randint(0, 1)
    hflip = randint(0, 1)

    def transform(x):
        nonlocal r90, hflip, vflip
        x = np.rot90(x)
        if hflip:
            x = x[:, ::-1]
        if vflip:
            x = x[::-1, :]
        return x.copy()
    return transform


def plot_all(*images, labels=None, swap_axes=False):
    _, ax = plt.subplots(1, len(images))
    for i, img in enumerate(images):
        if type(img) == torch.Tensor:
            img = img.cpu().detach()
        if len(img.shape) == 2:
            H, W = img.shape
            img = img.reshape(H, W, 1)
        if swap_axes:
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
        H, W, C = img.shape
        if C == 1:
            ax[i].imshow(img.reshape(H, W))
            if labels is not None:
                ax[i].set_title(labels[i])
        else:
            if labels is not None:
                ax[i].set_title(labels[i])
            ax[i].imshow(img)

def parse_tile(blob):
    x,y,z = blob
    x,y,z = float(x), float(y), int(z)
    return Tile(x,y,z)

def randrange(mag=0.5):
    return 1 + (mag - 2*mag*random())


class Sampler(Dataset):
    def __init__(self, test=False):
        super().__init__()
        self.data = []
        for file in Path("data/alignments").iterdir():
            with file.open() as f:
                self.data += json.load(f)
        #self.data = sorted(self.data, key=lambda x: parent(parse_tile(x['origin']), zoom=12))
        Random("seed").shuffle(self.data)
        self.split_point = int(len(self.data)*0.1)
        if test:
            self.data = self.data[:self.split_point]
        else:
            self.data = self.data[self.split_point:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = tile_engine[self.data[i]['origin']]
        raster_table = {
            "motorway": 2*randrange(0.3),
            "trunk": 10*randrange(0.3),
            "primary": 8*randrange(0.3),
            "residential": 3*randrange(0.3),
            "secondary": 5*randrange(0.3),
            "tertiary": 3*randrange(0.3),
            "unclassified": 2*randrange(0.3),
            "service": 2*randrange(0.3),
            "pedestrian": 1*randrange(0.3),
            "track": 2*randrange(0.3),
            "escape": 3*randrange(0.3),
            "footway": 1*randrange(0.3),
            "motorway_link": 8*randrange(0.3),
            "trunk_link": 4*randrange(0.3),
            "primary_link": 4*randrange(0.3),
            "secondary_link": 3*randrange(0.3),
            "tertiary_link": 3*randrange(0.3),
        }

        mask = road_engine.get_raster(self.data[i]['translation'], raster_table=raster_table)
        h,w = mask.shape
        mask = mask.reshape(h,w,1)
        t = make_transform()
        return t(image), t(mask)


if __name__ == "__main__":
    s = Sampler()
    img, mask = s[0]
    plot_all(img, mask)
    plt.show()
