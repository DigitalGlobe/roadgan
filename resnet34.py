import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from viz import viz
from common.nn_components import ResUNet34, Resnet34Discriminator
from common.losses import jaccard_loss, hybrid_loss, discriminator_loss, generator_loss, pack
from data.raster_data import Sampler, plot_all

from tqdm import tqdm
from random import randint
from pathlib import Path
import json
from os import rename

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPOCHS = 10
BATCH_SIZE = 10
REDRAW_INTERVAL = max(200//BATCH_SIZE, 1)

dataset = Sampler()
test_dataset = Sampler(test=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
test_loader = iter(DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=8))


def np_to_torch(x):
    if len(x.shape) == 3:
        h, w, c = x.shape
        x = x.reshape(1, h, w, c).copy()
    if type(x) == np.ndarray: 
        x = Tensor(x).cuda()
    if not x.is_cuda:
        x = x.cuda()
    return x.permute([0, 3, 1, 2]).float()

Path("models").mkdir(exist_ok=True, parents=True)
if Path("models/resnet34.pt").exists():
    print("Loading saved model...")
    generator = torch.load('models/resnet34.pt')
else:
    print("Initializing new generator...")
    generator = ResUNet34().cuda()

if Path('models/resnet34.json').exists():
    print("Found training state dict, will resume at correct iteration.")
    with Path('models/resnet34.json').open('r') as f:
        info_dict = json.load(f)
else:
    print("Did not find training state dict, may resume with incorrect loss weightings.")
    info_dict = {
        "iter": 0,
        "g_losses": [],
        "d_losses": [],
        "tr_j_scores": [],
        "te_j_scores": [],
        "g_batches": [],
        "d_batches": [],
        "tr_j_batches": [],
        "te_j_batches": []
    }


def make_r4_plot(img):
    c, h, w = img.shape
    img = img.cpu().numpy()
    new_arr = np.zeros((4, c, h, w))
    new_arr[0] = img
    new_arr[1] = img[:, ::-1, :]
    new_arr[2] = img[:, :, ::-1]
    new_arr[3] = img[:, ::-1, ::-1]
    img = Tensor(new_arr).cuda()
    img_pile = torch.sigmoid(generator(img)).cpu().numpy()
    img_pile[1] = img_pile[1, :, ::-1, :]
    img_pile[2] = img_pile[2, :, :, ::-1]
    img_pile[3] = img_pile[3, :, ::-1, ::-1]
    return img_pile.mean(0)


def adv_wf(x):
    return 1-np.exp(-x/50000)


g_optimizer = optim.Adam(params=generator.parameters(), lr=0.00001)

g_losses = info_dict['g_losses']
d_losses = info_dict['d_losses']
tr_j_scores = info_dict['tr_j_scores']
te_j_scores = info_dict['te_j_scores']
g_batches = info_dict['g_batches']
d_batches = info_dict['d_batches']
tr_j_batches = info_dict['tr_j_batches']
te_j_batches = info_dict['te_j_batches']

i = info_dict['iter']


for epoch in tqdm(range(EPOCHS), desc="Training"):
    even = True
    for img, mask in tqdm(loader, total=len(dataset)//BATCH_SIZE, desc="Epoch", leave=False):
        img = np_to_torch(img)
        mask = np_to_torch(mask)
        log_pred_mask = generator(img)
        pred_mask = torch.sigmoid(log_pred_mask)
        tr_j_scores += [float(jaccard_loss(pred_mask > 0.5, mask))]
        tr_j_batches += [i]
        
        g_loss = hybrid_loss(log_pred_mask, mask, pred_mask)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g_batches += [i]
        g_losses += [float(g_loss) if g_loss < 10 else 10]
        
        if i % REDRAW_INTERVAL == 1:
            with torch.no_grad():
                try: te_img, te_mask = next(test_loader)
                except: 
                    test_loader = iter(DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8))
                    te_img, te_mask = next(test_loader)
                te_img = np_to_torch(te_img)
                te_mask = np_to_torch(te_mask)
                log_te_pred_mask = generator(te_img)
                te_pred_mask = torch.sigmoid(log_te_pred_mask)
                te_j_scores += [float(jaccard_loss(te_pred_mask > 0.5, te_mask))]
                te_j_batches += [i]
                del te_img, te_mask, te_pred_mask
            with torch.no_grad():
                pred_mask = make_r4_plot(img[0])

            viz.push_images(img[0], mask[0], pred_mask, "-34")

            info_dict['iter'] = i+1
            with Path('models/resnet34.json').open('w+') as f:
                json.dump(info_dict, f)

            torch.save(generator, f"models/resnet34.pt.tmp")
            rename("models/resnet34.pt.tmp", "models/resnet34.pt")
        i += 1
        even = not even
