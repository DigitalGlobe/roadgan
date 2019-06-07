import visdom
import numpy as np

vis = visdom.Visdom()
w1 = vis.image(np.zeros((1, 512, 512)))
w2 = vis.image(np.zeros((1, 512, 512)))
w3 = vis.image(np.zeros((1, 512, 512)))
w4 = vis.image(np.zeros((1, 256, 384)))
w5 = vis.image(np.zeros((1, 256, 384)))
w6 = vis.image(np.zeros((1, 256, 384)))
w7 = vis.image(np.zeros((1, 256, 384)))
w8 = vis.text("Adversarial Weight:")

def push_images(basemap, r_mask, p_mask, taskel):
    vis.image(p_mask, opts={"title": f"Predicted Mask-{taskel}"}, win=w1)
    vis.image(r_mask, opts={"title": f"Real Mask-{taskel}"}, win=w2)
    vis.image(basemap, opts={"title": f"Basemap-{taskel}"}, win=w3)

def push_info_dict(info_dict, adv_loss):
    g_losses = info_dict['g_losses']
    d_losses = info_dict['d_losses']
    tr_j_scores = info_dict['tr_j_scores']
    te_j_scores = info_dict['te_j_scores']
    g_batches = info_dict['g_batches']
    d_batches = info_dict['d_batches']
    tr_j_batches = info_dict['tr_j_batches']
    te_j_batches = info_dict['te_j_batches']

    vis.text(f"Adversarial Weight: {adv_loss}", win=w8)
    if d_batches:
        vis.line(X=d_batches[-512:], Y=d_losses[-512:], win=w4,
                    opts={"title": "Discriminator Loss"})
    vis.line(X=g_batches[-512:], Y=g_losses[-512:], win=w5,
                opts={"title": "Generator Loss"})
    vis.line(X=tr_j_batches[-512:], Y=tr_j_scores[-512:], win=w6,
                opts={"title": "Jaccard Index (Train)"})
    vis.line(X=te_j_batches[-512:], Y=te_j_scores[-512:], win=w7,
                opts={"title": "Jaccard Index (Test)"})