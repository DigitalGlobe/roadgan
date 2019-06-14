import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from pathlib import Path
from random import shuffle, seed, randint, Random, random
from itertools import product

from scipy import ndimage as ndi
from skimage.draw import line_aa
from skimage.morphology import disk, closing, opening
from skimage.transform import resize


from tilelib import TileEngine, OSMRoadEngine

from loguru import logger
from tqdm import tqdm
from mercantile import children, parent, Tile
import json

tile_engine = TileEngine()
tile_engine.add_directory("data/tilestacks")

road_engine = OSMRoadEngine(postgis_connection_string="host=localhost user=rohit dbname=rohit password=bob")

def parse_tile(blob):
    x,y,z = blob
    x,y,z = float(x), float(y), int(z)
    return Tile(x,y,z)

def randrange(mag=0.5):
    return 1 + (mag - 2*mag*random())

def make_augmentation():
    r90 = randint(0, 3)
    vflip = randint(0, 1)
    hflip = randint(0, 1)
    smooth_opt = randint(0, 2)
    warp_colors = randint(0, 1)

    def flip_transform(x):
        nonlocal r90, hflip, vflip
        x = np.rot90(x)
        if hflip:
            x = x[:, ::-1]
        if vflip:
            x = x[::-1, :]
        return x.copy()

    def smooth(mask):
        nonlocal smooth_opt
        if smooth_opt == 0:
            return closing(mask, disk(randint(0,13)))
        elif smooth_opt == 1:
            return opening(mask, disk(randint(0,5)))
        else:
            return mask

    def photowarp(img):
        nonlocal warp_colors
        if not warp_colors:
            return img
        or_m, og_m, ob_m = r_m, g_m, b_m = np.mean(img, axis=(0,1))
        or_s, og_s, ob_s = r_s, g_s, b_s = np.std(img, axis=(0,1))
        r_m *= (0.5-random())/3+1
        g_m *= (0.5-random())/3+1
        b_m *= (0.5-random())/3+1
        r_s *= (0.5-random())/3+1
        g_s *= (0.5-random())/3+1
        b_s *= (0.5-random())/3+1

        img[:,:,0] = (img[:,:,0] - or_m) / or_s * r_s + r_m
        img[:,:,1] = (img[:,:,1] - og_m) / og_s * g_s + g_m
        img[:,:,2] = (img[:,:,2] - ob_m) / ob_s * b_s + b_m

        return np.clip(img, 0, 1)


    return lambda x: photowarp(flip_transform(x)), lambda x: smooth(flip_transform(x))

def slide(origin, translation):
    if random() < 0.25:
        dx, dy = random()*0.5, random()*0.5
        ox, oy, oz = origin
        tdx, tdy = random()*0.01, random()*0.01
        tx, ty, tz = translation
        return (ox+dx, oy+dy, oz), (tx+dx+tdx, ty+dy+tdy, tz)
    return origin, translation

class Sampler(Dataset):
    def __init__(self, test=False):
        super().__init__()
        self.data = []
        for file in Path("data/alignments").iterdir():
            with file.open() as f:
                self.data += json.load(f)


        Random("seed").shuffle(self.data)
        self.split_point = int(len(self.data)*0.1)
        if test:
            self.data = self.data[:self.split_point]
        else:
            self.data = self.data[self.split_point:]

        self.mapping_table = {i: i for i in range(len(self.data))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ip):
        i = self.mapping_table[ip]
        o_origin, o_translation = self.data[i]['origin'], self.data[i]['translation']
        origin, translation = slide(o_origin, o_translation)

        try:
            image = tile_engine[origin]
        except FileNotFoundError:
            origin, translation = o_origin, o_translation
            try:
                image = tile_engine[origin]
            except:
                logger.critical(f"Could not load data at value {ip} - remapping {ip} to {i+1}!")
                self.mapping_table[ip] = i+1
                return self[i+1]

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
        mask = road_engine.get_raster(translation, width_table=raster_table)
        t_i, t_m = make_augmentation()
        image, mask = t_i(image), t_m(mask)
        h, w  = mask.shape
        mask = mask.reshape(h, w, 1)
        return image, mask, self.data[i]['origin']

    def count_road_fraction(self):
        with_roads = 0
        without_roads = 0
        for ex in tqdm(self.data[:4096], desc="Road Sampler"):
            if road_engine[ex['translation']].sum() == 0:
                without_roads += 1
            else:
                with_roads += 1
        print(f"About {100*with_roads/4096}% of data tiles contain roads.")
        print(f"This translates to about {int(len(self)*with_roads/4096)} roaded tiles.")


if __name__ == "__main__":
    s = Sampler()
    s.count_road_fraction()
    #img, mask, tile = s[0]
    #plot_all(img, mask)
    #plt.show()
