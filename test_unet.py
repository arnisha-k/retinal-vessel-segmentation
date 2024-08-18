
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding
import torch.nn as nn
def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    
    mask = np.concatenate([mask, mask, mask], axis=-1)  
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("new_data/test/images/*"))
    test_y = sorted(glob("new_data/test/labels/*"))

    """ Hyperparameters """
    H = 592
    W = 592
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split("/")[-1].split(".")[0]

        """ Reading and padding image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for consistency
        image_padded = cv2.copyMakeBorder(image, top=0, bottom=592 - image.shape[0], left=0, right=592 - image.shape[1], borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image_tensor = torch.from_numpy(image_padded).float().permute(2, 0, 1) / 255.0
        x = image_tensor.unsqueeze(0).to(device)

        """ Reading and padding mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask_padded = cv2.copyMakeBorder(mask, top=0, bottom=592 - mask.shape[0], left=0, right=592 - mask.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
        mask_tensor = torch.from_numpy(mask_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
        y = mask_tensor.to(device)

        

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)


            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Visualizing and saving the result """
        # Convert pred_y back to the original shape for visualization if necessary
        
        pred_y_processed = mask_parse(pred_y)

        # Process original image for visualization
        image_vis = cv2.resize(image, (H, W))  # Ensure the image is resized back to 592x592 for consistent visualization
        image_vis_bgr = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
        ori_mask_vis = mask_parse(mask_padded)  # Use padded mask for visualization
        line = np.ones((size[1], 10, 3)) * 128  # Ensure line has the correct size

        cat_images = np.concatenate([image_vis_bgr, line, ori_mask_vis, line, pred_y_processed*255.0], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)
