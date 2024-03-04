"""
Description: Goal is to build a dataset based on the nuclei detection from nuclear segmentation and nucleus cell-type.
             Here we need a dataset that contains the nucleus label and the corresponding patch of the nucleus for
             training.

Requirements:
    - adata objects containing:
        - count matrix -> determine labels
        - nuclei boundaries -> extract the part of the image that will be useful
        - nucleus factors computed in nucleus_features
    - atlas: determine cell types from marker genes for the future

Implementations:
    [ x ]: select the adata for running the image
    [ x ]: build the labels for the adata objects
    [  ]: create a training and test datasets with 0.7 - 0.3
"""
import pickle
import random
# Std
from pathlib import Path
import os

import matplotlib.pyplot as plt
# Third Party
import numpy as np
import torch
import umap
from PIL import Image
from shapely import Polygon
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from matplotlib.lines import Line2D

# Relative imports
from src.utils import preprocess_transcriptomics, load_xenium_he_ome_tiff, get_results_path, get_human_breast_he_aligned_path
from src.nucleus_features.nucleus_features_extraction import create_masks_image

class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

def compute_ref_labels(adata: sc.AnnData, n_comp: int = 50, n_neighbors: int = 13):
    """ Leiden Clustering on the transcriptomcis contained inside the nucleus

        Remark: the filter will return a subset of the adata objects -> use cell_id for consistency
    """

    adata_ = adata.copy()
    sc.pp.pca(adata_, n_comps=n_comp)
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors)  # necessary for UMAP (k-neighbors with weights)
    sc.tl.umap(adata_)
    sc.tl.leiden(adata_)

    return adata_


def train_test(x, y):
    """ """
    split = StratifiedKFold(n_splits=5, random_state=62, shuffle=True)

    x, y = None


def create_polygon(x_, y_):
    """ Create Polygon form x_, y_ coordinate """
    stack = np.vstack((x_, y_)).T
    return Polygon(stack)


def extract_patch(img_he_, masks_, save_dir: Path, patch_size_=None):
    """ This function takes as input an image and return a list of patches corresponding to masks

        Strategy:
            - create masks image and then extend the min max of each nuclei -> issue with overlapping nuclei
            - create a polygon for each nucleus take the centroid and extract value
                x_centroid +/- value
                y_centroid +/- value
     """
    patches = []

    # Estimate patch size by taking the max size for each
    polygons = []
    max_size = 0
    size_distribution = []
    for id_ in np.unique(masks_.cell_id):
        coords = masks_[masks_.cell_id == id_]
        polygon = create_polygon(coords.vertex_x, coords.vertex_y)
        bounds = polygon.bounds
        size_ = max(bounds[2]-bounds[0], bounds[3]-bounds[1])
        max_size = size_ if size_ > max_size else max_size
        size_distribution.append(size_)
        polygons.append(polygon)

    if patch_size_ is None:
        patch_size_ = int(max_size)

    viz_count = 0
    viz_count_max = 50
    viz_dir = save_dir / "viz"
    os.makedirs(viz_dir, exist_ok=True)
    border_masks = []
    for ix, polygon in enumerate(polygons):
        min_x = int(polygon.centroid.x - patch_size_ // 2)
        max_x = int(polygon.centroid.x + patch_size_ // 2)
        min_y = int(polygon.centroid.y - patch_size_ // 2)
        max_y = int(polygon.centroid.y + patch_size_ // 2)

        if max_x - min_x != patch_size_:
            if max_x < img_he_.shape[1] - 1:
                max_x += 1
            else:
                min_x -= 1
        if max_y - min_y != patch_size_:
            if max_y < img_he_.shape[0] - 1:
                max_y += 1
            else:
                min_y -= 1

        if min_x < 0 or min_y < 0 or max_x >= img_he_.shape[1] or max_y >= image_he.shape[0]:
            border_masks.append(ix)
            continue

        # if min_x < 0:
        #     print("min_x < 0:", min_x)
        #     max_x -= min_x
        #     min_x = 0
        #
        # elif min_y < 0:
        #     print("min_y < 0:", min_y)
        #     max_y -= min_y
        #     min_y = 0
        # elif max_x >= img_he_.shape[1]:
        #     print(f"max_x > {img_he_.shape[1]}:", max_x)
        #     min_x += (img_he_.shape[1] - 1 - max_x)
        #     max_x = img_he_.shape[1] - 1
        #
        # elif max_y >= img_he_.shape[0]:
        #     print(f"max_y > {img_he_.shape[0]}:", max_y)
        #     min_y += (img_he_.shape[0] - 1 - max_y)
        #     max_y = img_he_.shape[0] - 1
        #
        # elif max_y <= min_y:
        #     print("inconsistent y")
        # elif max_x <= min_x:
        #     print("inconsistent x")

        patches.append(img_he_[min_y:max_y, min_x:max_x])

        if viz_count < viz_count_max and random.uniform(0, 1) > 0.9:
            try:
                viz_count += 1
                coords_ = np.array(polygon.exterior.coords)
                plt.plot(coords_[:, 0] - min_x, coords_[:, 1] - min_y)
                plt.imshow(img_he_[min_y:max_y, min_x:max_x])
                plt.savefig(viz_dir / f"visualization_{viz_count}.png")
                plt.close()
            except:
                print(min_x, max_x, min_y, max_y)

    plt.hist(size_distribution, fill=False, ec='r', bins=70)
    plt.xlabel("nucleus diameter [pixel]")
    plt.ylabel("counts")
    plt.savefig(save_dir / "diameter_distribution.png")
    plt.close()

    return np.array(patches), border_masks


def visualize_projection(patch, labels, save_dir_):

    cmap = plt.get_cmap("tab20")
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=5, label=label)
                       for i, label in enumerate(np.sort(np.unique(labels)))]

    # print("Starting Visualization")
    # patch_mean = np.mean(patch, axis=3)
    # patch_flatten = patch_mean.reshape((patch_mean.shape[0], patch_mean.shape[1]*patch_mean.shape[2]))
    # mean_ = np.mean(patch_flatten, axis=0)
    # std_ = np.std(patch_flatten, axis=0)
    # patch_normalized = (patch_flatten - mean_) / std_
    # patch_pca = PCA(n_components=50).fit_transform(patch_normalized)
    # for neigh in [5, 15, 25, 35, 50]:
    #     print(f"U-map {neigh}")
    #     embedding = umap.UMAP(n_neighbors=neigh, n_components=2).fit_transform(patch_pca)
    #     plt.figure()
    #     plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(labels), s=1)
    #     plt.xlabel("umap-1")
    #     plt.ylabel("umap-2")
    #
    #     plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
    #                title_fontsize="large")
    #     plt.savefig(save_dir_ / f"umap_fos_{neigh}.png")
    #     plt.close()
    #
    # # PCA
    # print("PCA")
    # embedding = PCA(n_components=2).fit_transform(patch_normalized)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(labels), s=1)
    # plt.xlabel("PCA-1")
    # plt.ylabel("PCA-2")
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
    #            title_fontsize="large")
    # plt.savefig(save_dir_ / f"pca_fos.png")
    # plt.close()

    print("Starting ResNet representation")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406 ], std=[0.229, 0.224, 0.225]),
    ])

    model = get_resnet_()
    dataset = CustomDataset(patch, transform=preprocess)
    data_loader = DataLoader(dataset, batch_size=32)

    representations = np.empty((0, 2048))
    progress_bar = tqdm(total=int(len(patch)/32), desc="Processing Batch")
    for img_tensor in data_loader:
        with torch.no_grad():
            representation = model(img_tensor).squeeze().detach().numpy()
        representations = np.vstack((representations, representation))
        progress_bar.update(1)
    progress_bar.close()

    embedding = PCA(n_components=2).fit_transform(representations)
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(labels), s=1)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
               title_fontsize="large")
    plt.savefig(save_dir_ / f"pca_fos_resnet50.png")
    plt.close()

    mean_ = np.mean(representations, axis=0)
    std_ = np.std(representations, axis=0)
    representations = (representations - mean_)/std_
    embedding = PCA(n_components=2).fit_transform(representations)
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(labels), s=1)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
               title_fontsize="large")
    plt.savefig(save_dir_ / f"pca_fos_resnet50_normalized.png")
    plt.close()

    patch_pca = PCA(n_components=50).fit_transform(representations)
    for neigh in [5, 15, 25, 35, 50]:
        embedding = umap.UMAP(n_neighbors=neigh, n_components=2).fit_transform(patch_pca)
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=cmap(labels), s=1)
        plt.xlabel("umap-1")
        plt.ylabel("umap-2")
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Category",
                   title_fontsize="large")

        plt.savefig(save_dir_ / f"umap_fos_resnet50_{neigh}.png")
        plt.close()

def get_resnet_():
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    layer_name = 'layer4'  # Example: 'layer4' for the last convolutional layer
    layer_index = list(resnet50.named_children()).index((layer_name, getattr(resnet50, layer_name)))
    layers = list(resnet50.children())[:layer_index + 2]
    return nn.Sequential(*layers)


def save_dataset(patch, label, save_dir: Path):
    dataset = [patch, label]
    with open(save_dir / "dataset.pkl", "wb") as file:
        pickle.dump(dataset, file)


def build_dataset(masks_adata_, img_he_, fractions_, save_path_: Path):
    """ """
    # Compute Labels
    cell_ids = masks_adata_.obs["cell_id"].copy()
    masks_adata_filtered = preprocess_transcriptomics(masks_adata_,)
    masks_adata_labels = compute_ref_labels(masks_adata_filtered)

    # Remove filtered cell's nucleus boundaries
    filtered_cell = set(cell_ids).difference(masks_adata_filtered.obs["cell_id"])
    filtered_index = [False if id_ in filtered_cell else True for id_ in masks_adata_filtered.uns["nucleus_boundaries"].cell_id]
    masks_adata_labels.uns["nucleus_boundaries_filtered"] = masks_adata_filtered.uns["nucleus_boundaries"][filtered_index]

    # Extract Corresponding patch
    patches, border_ix = extract_patch(img_he_, masks_adata_labels.uns["nucleus_boundaries_filtered"], save_path_, patch_size_=124)

    labels = masks_adata_labels.obs["leiden"].astype(int).to_numpy()
    labels = [label for ix, label in enumerate(labels) if ix not in border_ix]
    visualize_projection(patches, labels, save_path_)

    save_dataset(patches, labels, save_path_)


def load_dataset(dataset_name: str, sample: bool):
    return 0


if __name__ == "__main__":

    # --------------------- #
    # Run Parameters
    dataset_name = "stardist_qupath_leiden_clustering"
    fractions = [0.7, 0.15, 0.15]  # train, validation, test fraction
    # --------------------- #

    model = get_resnet_()

    save_path = get_results_path()
    scemila_data_path = save_path / "scemila" / dataset_name
    os.makedirs(scemila_data_path, exist_ok=True)

    # Tmp path for debugging
    masks_adata_path = Path("/Users/lbrunsch/Desktop/Phd/code/scratch/lbrunsch/results/nuclear_features/adata/Xenium-BreastCancer1-nucleus_stardist_qupath.h5ad")
    masks_adata = sc.read_h5ad(masks_adata_path)

    image_he_path = get_human_breast_he_aligned_path()
    image_he = load_xenium_he_ome_tiff(image_he_path, level_=0)

    build_dataset(masks_adata, image_he, fractions, scemila_data_path)

    x = load_dataset(dataset_name=dataset_name, sample=True)
