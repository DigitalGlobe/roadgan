import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from viz import viz
from common.nn_components import ReResUNet34, Resnet34Discriminator
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
BATCH_SIZE = 3
REDRAW_INTERVAL = max(200//BATCH_SIZE, 1)

dataset = Sampler()
test_dataset = Sampler(test=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
test_loader = iter(DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=4))


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
if Path("models/reresnet34-generator.pt").exists():
    print("Loading saved generator...")
    generator = torch.load('models/reresnet34-generator.pt')
else:
    print("Initializing new generator...")
    generator = ReResUNet34().cuda()

if Path('models/reresnet34-discriminator.pt').exists():
    print("Loading saved discriminator...")
    discriminator = torch.load('models/reresnet34-discriminator.pt')
else:
    print("Initializing new discriminator...")
    discriminator = Resnet34Discriminator().cuda()

if Path('models/reresnet34-SeGAN.json').exists():
    print("Found training state dict, will resume at correct iteration.")
    with Path('models/reresnet34-SeGAN.json').open('r') as f:
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
    img_pile = torch.sigmoid(generator.execute(img, 10)).cpu().numpy()
    img_pile[1] = img_pile[1, :, ::-1, :]
    img_pile[2] = img_pile[2, :, :, ::-1]
    img_pile[3] = img_pile[3, :, ::-1, ::-1]
    return img_pile.mean(0)


def adv_wf(x):
    return 1/(1+np.exp(-x/5000+4))

def rand_i(x):
    max_i = max(i//5000, 20)
    opts = np.arange(1+i//5000)
    p = np.exp(-opts)
    p /= p.sum()
    return max(max_i, np.random.choice(opts, p=p))

g_optimizer = optim.Adam(params=generator.parameters(), lr=0.00001)
d_optimizer = optim.Adam(params=discriminator.parameters(), lr=0.00001)

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
        log_pred_mask = generator.execute(img, rand_i(i))
        pred_mask = torch.sigmoid(log_pred_mask)
        tr_j_scores += [float(jaccard_loss(pred_mask > 0.5, mask))]
        tr_j_batches += [i]
        if even:
            pred_scores = discriminator(pack(pred_mask, img, binarize=True))
            g_loss = generator_loss(log_pred_mask, mask, pred_scores, adv_wf(i))
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            g_batches += [i]
            g_losses += [float(g_loss) if g_loss < 10 else 10]
        else:
            pred_scores = discriminator(pack(pred_mask.detach(), img))
            real_scores = discriminator(pack(mask, img))
            d_loss = discriminator_loss(pred_scores, real_scores)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            d_losses += [float(d_loss) if d_loss < 10 else 10]
            d_batches += [i]

        if i % REDRAW_INTERVAL == 1:
            with torch.no_grad():
                try: te_img, te_mask = next(test_loader)
                except: 
                    test_loader = iter(DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8))
                    te_img, te_mask = next(test_loader)
                te_img = np_to_torch(te_img)
                te_mask = np_to_torch(te_mask)
                log_te_pred_mask = generator.execute(te_img, rand_i(i))
                te_pred_mask = torch.sigmoid(log_te_pred_mask)
                te_j_scores += [float(jaccard_loss(te_pred_mask > 0.5, te_mask))]
                te_j_batches += [i]
                del te_img, te_mask, te_pred_mask
            with torch.no_grad():
                pred_mask = make_r4_plot(img[0])

            viz.push_images(img[0], mask[0], pred_mask)
            viz.push_info_dict(info_dict, adv_wf(i))

            info_dict['iter'] = i+1
            with Path('models/reresnet34-SeGAN.json').open('w+') as f:
                json.dump(info_dict, f)

            torch.save(generator, f"models/reresnet34-generator.pt.tmp")
            torch.save(discriminator, f"models/reresnet34-discriminator.pt.tmp")
            rename("models/reresnet34-generator.pt.tmp", "models/reresnet34-generator.pt")
            rename("models/reresnet34-discriminator.pt.tmp", "models/reresnet34-discriminator.pt")
        i += 1
        even = not even