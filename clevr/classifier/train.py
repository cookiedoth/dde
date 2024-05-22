# Adapted from https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/4d2cd3a422cc19e7a188c1c1ed9cde2a392aca92/classifier/train.py


# utility functions based off https://github.com/yilundu/improved_contrastive_divergence
# @article{du2020improved,
#   title={Improved Contrastive Divergence Training of Energy Based Models},
#   author={Du, Yilun and Li, Shuang and Tenenbaum, Joshua and Mordatch, Igor},
#   journal={arXiv preprint arXiv:2012.01316},
#   year={2020}
# }
import numpy as np

import os
import time
import copy
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import sys
from diffusion_cubes.classifier.model import Classifier

from torch.utils.data import Dataset, DataLoader

from diffusion_cubes.classifier.datasets import Clevr2DPosDataset

from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as tvu
from params_proto import ParamsProto
from ml_logger import logger
import torchinfo
from params_proto import Proto
from matplotlib import pyplot as plt
from diffusion_2d.utils import unlift

# seed
np.random.seed(301)
torch.manual_seed(301)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(301)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args(ParamsProto):
    spec_norm = True
    norm = True
    dataset = 'clevr_pos'
    lr = 1e-5
    batch_size = 32
    batch_display_size = 4
    im_size = 64
    num_epochs = 50
    batch_display_every = 100
    unet_dim_mults = (1, 2, 4)
    checkpoint_dir = 'results'
    data_path = Proto(env='$DATASET_ROOT/clevr/clevr_pos_data_128_30000.npz')

def display_batch(inputs, attr2, labels, logits_3d, batch_display_size, filename):
    batch_display_size = min(batch_display_size, inputs.shape[0])
    _, ax = plt.subplots(batch_display_size, 3, figsize=(10, 5 * batch_display_size))
    for i in range(batch_display_size):
        ax[i, 0].imshow(unlift(inputs[i].permute(1, 2, 0)))

        im1 = torch.ones_like(inputs[i])
        set0 = [1, 2] if labels[i].item() else [0, 1]
        for ch_id in set0:
            im1[ch_id, attr2[i, 0].item(), attr2[i, 1].item()] = 0.0
        ax[i, 1].imshow(unlift(im1.permute(1, 2, 0)))

        ax[i, 2].imshow(unlift(logits_3d[i]))
    plt.tight_layout()
    logger.savefig(filename)

def train_model(
        model, dataloaders, optimizer, start_epoch=0, num_epochs=50
):
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        logger.print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_index = 0

            # Iterate over data.
            for i, (inputs, attr, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                attr = attr.float().to(device)

                # Do the transformation to pixel
                im_size = inputs.shape[2]
                attr2 = torch.zeros_like(attr)
                attr2[:, 0] = 1.0 - attr[:, 1]
                attr2[:, 1] = attr[:, 0]
                attr2 *= im_size
                attr2 = torch.round(attr2).long()
                attr2 = torch.clip(attr2, min=0, max=im_size - 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val':
                        model.eval()
                    else:
                        model.train()

                    inputs_cnt = labels.shape[0]
                    logits_3d = model(inputs)
                    logits_1d = torch.zeros_like(labels)
                    logits_1d = logits_3d[torch.arange(inputs_cnt), attr2[:, 0], attr2[:, 1]]

                    if i % Args.batch_display_every == 0:
                        print(f'Plotting visualization for epoch {epoch}, phase {phase}, step {i}')
                        display_batch(inputs, attr2, labels, logits_3d, Args.batch_display_size, f'figures/batch_{epoch}_{phase}_{i}.png')

                    loss = criterion(sigmoid(logits_1d), labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    preds = (logits_1d > 0.5)
                    corrects = torch.sum(preds == labels) / labels.shape[0]

                    running_loss += loss.item()
                    running_corrects += corrects.item()
                    running_index += 1

            epoch_loss = running_loss / running_index
            epoch_acc = running_corrects / running_index
            logger.print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                logger.store_metrics(val_loss=epoch_loss)
                logger.store_metrics(val_acc=epoch_acc)
            else:
                logger.store_metrics(train_loss=epoch_loss)
                logger.store_metrics(train_acc=epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.print(f'Saving the checkpoint at epoch {epoch}', color='yellow')
                logger.torch_save(model, f'checkpoints/model_best.pkl')
        logger.log_metrics_summary(key_values={'epoch': epoch})

    time_elapsed = time.time() - since
    logger.print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.print('Best val Acc: {:4f}'.format(best_acc))

    logger.torch_save(model, f'checkpoints/model_last.pkl')
    model.load_state_dict(best_model_wts)
    return model


def main(**kwargs):
    Args._update(kwargs)

    print(logger)
    logger.log_text("""
       charts:
       - yKey: train_loss/mean
         xKey: epoch
       - yKey: train_acc/mean
         xKey: epoch
       - yKey: val_loss/mean
         xKey: epoch
       - yKey: val_acc/mean
         xKey: epoch
       - type: image
         glob: figures/batch_0_val_0.png
       - type: image
         glob: figures/batch_5_val_0.png
       - type: image
         glob: figures/batch_10_val_0.png
       - type: image
         glob: figures/batch_15_val_0.png
       - type: image
         glob: figures/batch_19_val_0.png""", '.charts.yml', dedent=True, overwrite=True)

    model = Classifier(dim=Args.im_size, dim_mults=Args.unet_dim_mults)
    model = model.train().to(device)
    torchinfo.summary(model)

    optimizer = optim.Adam(model.parameters(), lr=Args.lr, betas=(0.9, 0.999), eps=1e-8)
    datasets = {phase: Clevr2DPosDataset(data_path=Args.data_path, resolution=Args.im_size, split=phase) for phase in ['train', 'val']}

    dataloaders = {phase: DataLoader(
        dataset=datasets[phase], shuffle=True, pin_memory=True, num_workers=4, batch_size=Args.batch_size)
        for phase in ['train', 'val']
    }

    train_model(model, dataloaders, optimizer, 0, Args.num_epochs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--spec_norm", action="store_true", default=True)
    parser.add_argument("--norm", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--im_size", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=150)

    args = parser.parse_args()

    kwargs = dict(
        spec_norm=args.spec_norm,
        norm=args.norm,
        lr=args.lr,
        batch_size=args.batch_size,
        im_size=args.im_size,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs
    )

    main(**kwargs)
