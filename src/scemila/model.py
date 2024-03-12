import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.utils import check_gpu


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma * out + x
        return out


class CNNClassifierWithAttention(nn.Module):
    def __init__(self, num_classes, in_dim, conv_type="scracth"):

        super(CNNClassifierWithAttention, self).__init__()

        self.conv_type = conv_type
        if self.conv_type == "conv":
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.num_feature_maps = 128
            self.self_attention = SelfAttention(self.num_feature_maps)
            self.pool = nn.MaxPool2d(2, 2)
        elif conv_type.lower() == "resnet":
            self.conv = models.resnet50(pretrained=True)
            self.num_feature_maps = 2048
            self.self_attention = SelfAttention(self.num_feature_maps)
            for param in self.conv.parameters():
                param.requires_grad = False
            # Unfreeze last layer
            for param in self.conv.layer4.parameters():
                param.requires_grad = True
        elif conv_type.lower() == "vision-transformer":
            pass

        # Determine the output size after conv layers
        with torch.no_grad():
            x = torch.zeros(1, 3, in_dim, in_dim)
            if self.conv_type == "conv":
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.self_attention(F.relu(self.conv3(x)))
                self.feature_maps_dim = x.size(2)

            elif self.conv_type == "resnet":
                x = self.resnet_forward(x)
                x = self.self_attention(x)
                self.feature_maps_dim = x.size(2)

        self.fc1 = nn.Linear(self.num_feature_maps * self.feature_maps_dim * self.feature_maps_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def resnet_forward(self, x):
        x = self.conv.conv1(x)
        x = self.conv.bn1(x)
        x = self.conv.relu(x)
        x = self.conv.maxpool(x)

        x = self.conv.layer1(x)
        x = self.conv.layer2(x)
        x = self.conv.layer3(x)
        return self.conv.layer4(x)

    def forward(self, x):

        if self.conv_type == "conv":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))  # add adaptative pooling here ?

        elif self.conv_type.lower() == "resnet":
            x = self.resnet_forward(x)

        x = self.self_attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    # Load DataSet
    dataset_path = "/Users/lbrunsch/Desktop/Phd/code/scratch/lbrunsch/results/scemila/stardist_qupath_he_dapi_match_leiden_clustering/dataset.pkl"
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

    device = check_gpu()
    model = CNNClassifierWithAttention(num_classes=10, in_dim=96, conv_type="resnet").to(device)

    print(model)
    example = torch.Tensor(np.transpose(dataset[0][0], axes=(2, 0, 1))).unsqueeze(0).to(device)

    output = model(example)
