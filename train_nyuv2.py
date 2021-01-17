import time
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
import numpy as np

from models.DenseDepth import DenseDepth

from losses import SSIM
from dataloader.nyuv2 import getTrainingTestingData
from utils.scheduler import CosineAnnealingLR
import configs.configs as config

import warnings
warnings.filterwarnings("ignore")


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def train(epoch, trainloader, optimizer, model, scheduler, device):
    model.train()

    avg_loss = 0
    avg_dog_loss = 0
    avg_grad_loss = 0

    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    ssim_criterion = SSIM()

    for idx, batch in enumerate(trainloader):
        global_step = epoch * len(trainloader) + idx

        optimizer.zero_grad() 

        image = torch.Tensor(batch["image"]).to(device)
        depth = torch.Tensor(batch["depth"]).to(device)

        pred_depth = model(image)

        # calculating the losses 
        l1_loss = l1_criterion(pred_depth, depth)
        ssim_loss = 1 - ssim_criterion(depth, pred_depth)

        loss = (1.0 * ssim_loss) + (0.1 * torch.mean(l1_loss))
        net_loss = loss

        avg_loss += loss.item()

        net_loss.backward()
        optimizer.step()

        scheduler.step(global_step)

        # Logging  
        num_iters = epoch * len(trainloader) + idx  
        if (idx % config.TRAIN_NYUV2['print_freq'] == 0) and (idx > 0):
            avg_loss = avg_loss / config.TRAIN_NYUV2['print_freq']

            print(
                "Epoch: #{0} Batch: {1}/{2}\t"
                "Lr: {lr:.6f}\t"
                "LOSS: {loss:.4f}\t"
                .format(epoch, idx, len(trainloader), lr=optimizer.param_groups[-1]['lr'], loss=avg_loss)
            )

            avg_loss = 0

def valid(testloader, model, device):
    model.eval()

    err = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    err = np.array(err)
    err_list = ["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image = torch.Tensor(batch["image"]).to(device)
            depth = torch.Tensor(batch["depth"]).to(device)

            pred_depth = model(image)

            depth = (depth.detach().cpu().numpy())
            pred_depth = (pred_depth.detach().cpu().numpy())

            pred_depth[pred_depth < 1e-1] = 1e-1
            pred_depth[pred_depth > 10] = 10
            pred_depth[np.isinf(pred_depth)] = 10
            pred_depth[np.isnan(pred_depth)] = 1e-1

            valid_mask = np.logical_and(depth > 1e-1, depth < 10)

            err += compute_errors(depth[valid_mask], pred_depth[valid_mask])

        err = err / (idx + 1)

        for i in range(len(err)):
            print(err_list[i], ":",  err[i])
        
        return err[3]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_NYUV2['device']

    print("Load dataset...")
    trainloader, testloader = getTrainingTestingData(config.DATA['NYUV2_data_root'], batch_size=config.TRAIN_NYUV2['batch_size'])
    print("ok!")

    print("Check device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("ok!")

    print("Load model...")
    model = DenseDepth(max_depth=10, encoder_pretrained=config.TRAIN_NYUV2['pretrain'])
    model = model.to(device)
    print("ok!")

    print("Load optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_NYUV2['learning_rate'])
    print("ok!")

    # Starting training 
    print("Starting training... ")

    scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader)*config.TRAIN_NYUV2['epoch'], eta_min=0.00001, warmup=None, warmup_iters=None)

    best = 1
    
    for epoch in range(config.TRAIN_NYUV2['epoch']):
        train(epoch, trainloader, optimizer, model, scheduler, device)
        rmse = valid(testloader, model, device)

        if rmse < best:
            torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict()
            }, "ckpt_{}.pth".format(epoch))

            best = rmse


if __name__ == "__main__":
    main()