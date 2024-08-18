
import os
import time
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
#from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from model_unetr import UNETR

def l1_regularization(model, lambda_l1):
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    return lambda_l1 * l1_reg

def l2_regularization(model, lambda_l2):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)**2
    return lambda_l2 * l2_reg



def train(model, loader, optimizer, loss_fn, device, lambda_l1, lambda_l2):
    epoch_loss = 0.0

    model.train()
    # Initialize tqdm progress bar
    progress_bar = tqdm(loader, desc='Training', total=len(loader))
    for x, y in progress_bar:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        l1_reg_loss = l1_regularization(model, lambda_l1)
        l2_reg_loss = l2_regularization(model, lambda_l2)
        total_loss = loss + l1_reg_loss + l2_reg_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    # Initialize tqdm progress bar
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc='Evaluating', total=len(loader))
    model.eval()
    with torch.no_grad():
        for x, y in progress_bar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss
    
lambda_l1 = 1e-5
lambda_l2 = 1e-4

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("new_data/train/images/*"))
    train_y = sorted(glob("new_data/train/labels/*"))
    
    
    valid_x = sorted(glob("new_data/test/images/*"))
    valid_y = sorted(glob("new_data/test/labels/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 592
    W = 592
    size = (H, W)
    batch_size = 2
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda')   ## GTX 1060 6GB
    model = UNETR(in_channels = 3, out_channels = 1, img_size = (592,592))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device, lambda_l1, lambda_l2)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
