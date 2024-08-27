import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from Models.hptModel import FrequencySeparationNet,ResNet

def eval_model(path):
    model.load_state_dict(torch.load(path))
    model.eval()

    def evaluate(loader, mode='train'):
        correct = 0
        total = 0
        all_outmid = []
        all_y = []
        TP = 0  # True Positive
        TN = 0  # True Negative
        FP = 0  # False Positive
        FN = 0  # False Negative

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output, outmid = model(X_batch)
                all_outmid.append(outmid)
                all_y.append(y_batch)
                _, predicted = torch.max(output, dim=1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                TP += ((predicted == 1) & (y_batch == 1)).sum().item()
                TN += ((predicted == 0) & (y_batch == 0)).sum().item()
                FP += ((predicted == 1) & (y_batch == 0)).sum().item()
                FN += ((predicted == 0) & (y_batch == 1)).sum().item()

        accuracy = correct / total
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        false_negative_rate = FN / (FN + TP) if FN + TP > 0 else 0
        false_positive_rate = FP / (FP + TN) if FP + TN > 0 else 0

        print(f'{mode.capitalize()} correct:', correct)
        print(f'{mode.capitalize()} total:', total)
        print(f'{mode.capitalize()} Accuracy: {accuracy}')
        print(f'{mode.capitalize()} Precision: {precision}')
        print(f'{mode.capitalize()} Recall: {recall}')
        print(f'{mode.capitalize()} F1 Score: {f1_score}')
        print(f'{mode.capitalize()} False Negative Rate: {false_negative_rate}')
        print(f'{mode.capitalize()} False Positive Rate: {false_positive_rate}')

    evaluate(train_loader, mode='train')
    evaluate(test_loader, mode='test')

if __name__ == '__main__':
    path = 'path_you_save_model'
    train_loader, test_loader, num_classed = create_bcg_dataset()

    # 检查GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    FSnet = FrequencySeparationNet().to(device)
    model = ResNet(1, 2, [100, 50, 25],FSnet,500).to(device)

    eval_model(path)

