import numpy as np
from matplotlib import pyplot as plt

import cv2
import tifffile as tiff
from skimage.morphology import closing, medial_axis, disk
import sknw
from shapely.geometry import LineString
import geopandas as gpd
from osgeo import gdal

from loguru import logger
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

import torch
from torch import Tensor

from mercantile import bounds, Tile, tile, parent
from tilelib import OSMRoadEngine, TileEngine

from pathlib import Path

from sys import argv

image_directory = Path(argv[1])
assert image_directory.exists(), "Path must exist!"
assert image_directory.is_dir(), "File must be a directory!"


model = torch.load("models/resnet34-generator.pt")

def read_image(file):
    logger.info(f"Opening {file}...")
    file = str(file.absolute())
    try:
        img = tiff.imread(file)
    except Exception as e:
        logger.critical(f"{file} is corrupt or otherwise inaccessible! Returning black 1024x1024 image...")
        return np.zeros((1024,1024,3))
    logger.info(f"Tonemapping {file}...")
    img_colors = np.array([x for x in img.reshape(-1, 3) if x.sum()>0])
    lo = np.array([0,0,0])
    hi = np.array([0,0,0])
    for i in range(3):
        try:
            lo[i] = np.percentile(img_colors[:,i], 2)
            hi[i] = np.percentile(img_colors[:,i], 98)
        except Exception as e:
            logger.critical(f"{file} is all black on channel {i}! Refusing to tonemap...")
            lo[i] = 0
            hi[i] = 1
    img = (img - lo) / (hi-lo)
    img = np.clip(img, 0, 1)
    return img

def np_to_torch(x):
    if len(x.shape) == 3:
        h, w, c = x.shape
        x = x.reshape(1, h, w, c).copy()
    if type(x) == np.ndarray: 
        x = Tensor(x).cuda()
    if not x.is_cuda:
        x = x.cuda()
    return x.permute([0, 3, 1, 2]).float()

def make_d8_plot(img):
    logger.info(f"Segmenting image...")

    h, w, c = img.shape
    new_arr = np.zeros((7, c, h, w))
    c_img = img.swapaxes(1,2).swapaxes(0,1)
    new_arr[0] = c_img
    new_arr[1] = c_img[:, ::-1, :]
    new_arr[2] = c_img[:, :, ::-1]
    new_arr[3] = c_img[:, ::-1, ::-1]
    new_arr[4] = np.rot90(img, k=1).swapaxes(1,2).swapaxes(0,1)
    new_arr[5] = np.rot90(img, k=2).swapaxes(1,2).swapaxes(0,1)
    new_arr[6] = np.rot90(img, k=3).swapaxes(1,2).swapaxes(0,1)
    img = Tensor(new_arr).cuda()
    img_pile = np.zeros((7, 1, h ,w))
    for i in range(7):
        img_pile[i, 0, :, :] = torch.sigmoid(model(img[i:i+1])).cpu().numpy()
    
    img_pile[1] = img_pile[1, :, ::-1, :]
    img_pile[2] = img_pile[2, :, :, ::-1]
    img_pile[3] = img_pile[3, :, ::-1, ::-1]
    img_pile[4] = np.rot90(img_pile[4,:,:,:].swapaxes(0,1).swapaxes(1,2), k=-1).swapaxes(1,2).swapaxes(0,1)
    img_pile[5] = np.rot90(img_pile[5,:,:,:].swapaxes(0,1).swapaxes(1,2), k=-2).swapaxes(1,2).swapaxes(0,1)
    img_pile[6] = np.rot90(img_pile[6,:,:,:].swapaxes(0,1).swapaxes(1,2), k=-3).swapaxes(1,2).swapaxes(0,1)

    return img_pile.mean(0)

def segment(basemap):
    delta = (1024-780)//2
    def make_nn_input(basemap):
        basemap = basemap.copy()
        basemap = cv2.resize(basemap, (780, 780))
        basemap = cv2.copyMakeBorder(basemap,delta,delta,delta,delta,cv2.BORDER_REFLECT)
        return basemap
    basemap = make_nn_input(basemap)
    with torch.no_grad():
        mask = make_d8_plot(basemap)
    mask = mask[:, delta:-delta, delta:-delta]
    return mask

def skeletonize(mask):
    m_img = mask[0,:,:] > 0.5/8
    m_img = closing(m_img)
    skel, dist = medial_axis(m_img, return_distance=True)
    return skel, dist

def get_geotiff_bbox(file):
    file = str(file.absolute())
    src = gdal.Open(file)
    if src is None:
        return (0,0),(1,1)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    return (uly, ulx), (lry, lrx)

def vectorize(blob):
    (skel, dist), path = blob

    logger.info(f"Vectorizing {path}...")
    graph = sknw.build_sknw(skel)

    df = gpd.GeoDataFrame(columns=['geometry', 'width'])

    ul, lr = get_geotiff_bbox(path)
    dy = ul[0]-lr[0]
    dx = lr[1]-ul[1]

    for s,e in graph.edges():
        H, W = dist.shape
        widths = [dist[tuple(pt)] for pt in graph[s][e]['pts']]
        width = np.percentile(widths, 33)/H*dy
        N = len(graph[s][e]['pts'])+2
        pts = np.zeros((N,2), dtype=np.float64)
        pts[0] = graph.node[s]['o']
        pts[-1] = graph.node[e]['o']
        pts[1:-1] = graph[s][e]['pts']
        
        pts[:, 0] = 1-pts[:, 0]/H
        pts[:, 0] *= dy
        pts[:, 0] += lr[0]
        
        pts[:, 1] = pts[:, 1]/W
        pts[:, 1] *= dx
        pts[:, 1] += ul[1]
        
        df = df.append({
            'geometry': LineString(pts[:, ::-1]),
            'width': width
        }, ignore_index=True)
    return df

p = Pool(cpu_count()-1)


paths = [f for f in image_directory.iterdir() if "tif" in f.name]
imgs = p.imap(read_image, paths)
masks = map(segment, imgs)
skels = map(skeletonize, masks)
dfs = map(vectorize, zip(skels, paths))

for df, path in tqdm(zip(dfs, paths)):
    name = path.name[:-4]+".geojson"
    if not df.empty:
        df.to_file(image_directory/name, driver="GeoJSON")
