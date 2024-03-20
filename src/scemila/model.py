import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.utils import check_gpu


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes, model_type, in_dim, attention_layer, unfrozen_layers):
        super(ImageClassificationModel, self).__init__()

        if model_type == "cnn":
            self.model = CNNClassifierWithAttention(num_classes, in_dim, attention_layer)
        elif model_type == "resnet":
            resnet_model = "resnet50"
            self.model = ResNetAttention(num_classes, in_dim, resnet_model, attention_layer)
        elif "vit" in model_type:
            self.model = VisionTransformer(num_classes, model_type, unfrozen_layers)
        else:
            raise ValueError("Not a model type choose in [cnn, resnet, vit]")

    def forward(self, x):
        return self.model(x)

    def extract_feature(self, layer_name):
        self.model.extract_feature(layer_name)

    def save(self, dir_):
        torch.save(self.state_dict(), dir_ / "model_parameters.pth")

    def load(self, dir_):
        state_dict = torch.load(dir_ / "model_parameters.pth", map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)


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


class VisionTransformer(nn.Module):
    def __init__(self, num_classes, model_type, unfrozen_layers):
        super(VisionTransformer, self).__init__()

        assert model_type in ["vit_16", "vit_32"]

        self.model_type = model_type
        self.n_unfrozen_layers = unfrozen_layers

        if model_type == "vit_16":
            self.backbone = models.vit_b_16(weights="DEFAULT")
            # Freeze the network
            for parameters in self.backbone.parameters():
                parameters.requires_grad = False

            for params in self.backbone.encoder.ln.parameters():
                params.requires_grad = True

            encoder_layers = [f"encoder_layer_{i}" for i in range(11, -1, -1)]
            layers_to_unfreeze = encoder_layers[0:self.n_unfrozen_layers]
            for layer in layers_to_unfreeze:
                for params in getattr(self.backbone.encoder.layers, layer).parameters():
                    params.requires_grad = True

        else:
            self.backbone = models.vit_b_32(weights="DEFAULT")

        # Replace classification head
        linear_layers = [nn.Linear(self.backbone.heads.head.in_features, 512),
                         nn.BatchNorm1d(512),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(512, 256),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(256, num_classes)]
        self.backbone.heads = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.backbone(x)

    def extract_feature(self, layer_name, hook):
        self.backbone.encoder.get_laregister_forward_hook(hook)


class ResNetAttention(nn.Module):
    def __init__(self, num_classes, in_dim, resnet_model="resnet50", attention_layer=True):
        super(ResNetAttention, self).__init__()

        assert resnet_model in ["resnet50"]

        if resnet_model == "resnet50":
            self.conv = models.resnet50(pretrained=True)
            self.num_feature_maps = 2048
            for param in self.conv.parameters():
                param.requires_grad = False
            # Unfreeze last layer
            for param in self.conv.layer4.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Not Implemented")

        if attention_layer:
            self.self_attention = SelfAttention(self.num_feature_maps)
        else:
            self.self_attention = nn.Identity()

        # Determine the output size after conv layers
        with torch.no_grad():
            x = torch.zeros(1, 3, in_dim, in_dim)
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
        x = self.resnet_forward(x)
        x = self.self_attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNNClassifierWithAttention(nn.Module):
    def __init__(self, num_classes, in_dim, attention_layer=True):

        super(CNNClassifierWithAttention, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.num_feature_maps = 128
        if attention_layer:
            self.self_attention = SelfAttention(self.num_feature_maps)
        else:
            self.self_attention = nn.Identity()

        # Determine the output size after conv layers
        with torch.no_grad():
            x = torch.zeros(1, 3, in_dim, in_dim)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.self_attention(F.relu(self.conv3(x)))
            self.feature_maps_dim = x.size(2)

        self.fc1 = nn.Linear(self.num_feature_maps * self.feature_maps_dim * self.feature_maps_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))  # add adaptative pooling here ?

        x = self.self_attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


if __name__ == "__main__":

    device = check_gpu()
    model = ImageClassificationModel(num_classes=10, model_type="vit_16", in_dim=96, attention_layer=True)

    print(model)

    example = torch.zeros(1, 3, 224, 224).to(device)
    model.to(device)

    output = model(example)

    print(output)
