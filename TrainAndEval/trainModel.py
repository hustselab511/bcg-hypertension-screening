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

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from Models.hptModel import FrequencySeparationNet,ResNet


def hierarchical_loss(tensor):
    data = tensor.cpu().detach().numpy()

    dist_matrix = pdist(data, metric='euclidean')

    linkage_matrix = sch.linkage(dist_matrix, method='average')

    assignments = sch.fcluster(linkage_matrix, t=2, criterion='maxclust')

    centroids = []
    for i in range(1, 3):
        cluster_points = data[assignments == i]
        centroid = torch.tensor(cluster_points.mean(axis=0), dtype=tensor.dtype, device=tensor.device)
        centroids.append(centroid)

    centroids = torch.stack(centroids)

    assignments = torch.tensor(assignments - 1, dtype=torch.long, device=tensor.device)

    loss = torch.mean((tensor - centroids[assignments]) ** 2)

    return loss


def train_model(model_path='checkpoint.pt'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    train_losses = []
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in range(num_epoches):
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for X_batch, y_batch in train_bar:
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output, outmid = model(X_batch)

            mid_loss = 2*hierarchical_loss(outmid)
            loss = criterion(output, y_batch) + mid_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        model.eval()

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        valid_losses = []

    torch.save(model.state_dict(), model_path)
    print('模型保存完毕！')
    return model




if __name__ == '__main__':
    num_epoches = 100
    batch_size = 256
    path = 'path_to_save_model'
    train_loader, test_loader, num_classed = create_bcg_dataset()

    # 检查GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    FSnet = FrequencySeparationNet().to(device)
    FSnet.load_state_dict(torch.load('path_you_save_pre-training_FSnet'))
    model = ResNet(1, 2, [100, 50, 25],FSnet,500).to(device)

    model_train = train_model(path)

