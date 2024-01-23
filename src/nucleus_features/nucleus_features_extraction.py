"""
Description: This python file has the purpose to extract nuclei features based on nuclear seg. The goal is to use the
             segmentation information provided by different tools such as CellPose, Stardist, CellProfiler and
             compute different features to see how one  can classify nucleus based on nuclear morphology features.

Author: Lucas Brunschwig (lucas.brunschwig@epfl.ch)

Development:
[ x ](1)Choose an algorithm and a type of image to extract the nucleus information
        - strategy 1: start with 2D clustering algorithm
            - using CellPose
        - strategy 2: build a good performing 3D algorithm

[  ](2) Find out how to extract nucleus individually for analysis
        - strategy 1: store masks.pkl (containing the label for each mask)
                      can be used to extract the pixel for each mask
        - strategy 2

Last Revision: 18.01.23
"""
import os
import platform
# Std
from pathlib import Path
from typing import Optional

# Third party
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyfeats import fos
from scipy import stats

# Relative import
from ..utils import load_image, load_xenium_data, image_patch
from ..nucleus_segmentation.segmentation_watershed import segment_watershed
from src.nucleus_segmentation.segmentation_cellpose import segment_cellpose
from src.nucleus_segmentation.segmentation_stardist import segment_stardist
from src.nucleus_segmentation.utils import get_xenium_nucleus_boundaries, get_masks


SEGMENTATION = {"cellpose": segment_cellpose, "stardist": segment_stardist, "watershed": segment_watershed}

if platform.system() == "Linux":
    WORKING_DIR = Path("..")
else:
    WORKING_DIR = Path("../../..")

RESULTS = Path()


def load_masks(method_: str, path_replicate_: Path = Path(), square_size_: Optional[int] = None,
               img_type_: str = "mip", compute_masks: bool = False):
    """ This method will load masks based on methods and specific parameters """

    if method_ == "xenium":
        return get_xenium_nucleus_boundaries(path_replicate_=path_replicate_)

    # Select Masks with Optimized parameters
    param = {}
    if method_ == "cellpose":
        param = {"model_": "cyto", "diameter_": 30}
    elif method == "stardist":
        param = {"prob_thrsh": 0.3, "nms_thrsh": 0.5}
    elif method == "watershed":
        param = {}

    # Compute Masks or Load Masks
    # TODO: simplify this by testing if the file exists in the segmentation method
    if compute_masks:
        return SEGMENTATION[method_]()
    else:
        return get_masks(method, param, square_size_=square_size_, img_type_=img_type_)


def compute_fos(flattened_signal) -> tuple:

    features = [np.mean(flattened_signal),
                np.median(flattened_signal),
                np.std(flattened_signal),
                stats.skew(flattened_signal),
                stats.kurtosis(flattened_signal),
                min(flattened_signal),
                max(flattened_signal),
                np.percentile(flattened_signal, 10),
                np.percentile(flattened_signal, 25),
                np.percentile(flattened_signal, 75),
                np.percentile(flattened_signal, 90)]

    features_name = ["mean", "median", "std", "skewness", "kurtosis", "min", "max", "q10", "q25", "q75", "q90"]

    return features, features_name


def get_cell_label(path_replicate_: Path, label_name_: str, masks: np.ndarray):
    """ This will require some work to recover cell label (obtained through transcriptomics)
        with their location on the plane """

    adata = load_xenium_data(path_replicate_)

    # Two Strategy:
    # - take previous labels from graph clustering and find overlapping nucleus boundaries
    # - compute cell boundaries based on new nucleus boundaries and perform cell transcriptomics association
    # Second Strategy seems better since it avoids issues with nucleus boundaries that are extremely different

    return adata.obs[label_name_].astype(int).tolist()


def get_cell_area(dapi_signal):

    cell_per_micron = 0.2125
    return len(dapi_signal) * (cell_per_micron ** 2)


def run_nucleus_features_extraction(path_replicate_: Path, image_type_: str, level_: int,
                                    square_size_: Optional[int], method_: str):

    # Load Image
    img = load_image(path_replicate_, image_type_, level_)
    patch, boundaries = image_patch(img, square_size_)

    # Compute Masks
    masks = load_masks(method_, path_replicate_, square_size_, image_type_)

    labels = get_cell_label(path_replicate_, label_name_="graph_clusters", masks=masks)
    cmap = plt.get_cmap("tab20c")
    labels_color = [cmap(i) for i in labels]

    # Compute Nucleus Features
    nucleus_features = []

    # Compute Masks features
    print("Extracting - nuclear features")

    progress_bar = tqdm(total=len(np.unique(masks)), desc="Processing")

    for label in np.unique(masks)[1:500]:

        masks_copy = masks.copy()
        masks_copy[np.where(masks != label)] = 0
        masks_copy[np.where(masks == label)] = 1

        # First order statistic
        mask_coordinate = np.where(masks == label)
        dapi_signal = patch[mask_coordinate]
        fos_value, fos_name = compute_fos(dapi_signal)
        cell_area = get_cell_area(dapi_signal)
        nucleus_features.append(fos_value + [cell_area])

        progress_bar.update(1)

    # Visualize U-map with first order-moment
    nucleus_features = np.array(nucleus_features)

    embedding = umap.UMAP(n_neighbors=15, n_components=2).fit_transform(nucleus_features)

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_color)
    plt.xlabel("umap-1")
    plt.ylabel("umap-2")
    plt.savefig(RESULTS / f"umap_fos_{image_type_}_{square_size_}_{method_}.png")

    return 0


def build_dir():
    global RESULTS
    RESULTS = WORKING_DIR / "scratch/lbrunsch/results/nuclear_features"
    os.makedirs(RESULTS, exist_ok=True)


if __name__ == "__main__":

    # setup
    build_dir()

    # Image Parameters
    replicates_dir = WORKING_DIR / "scratch/lbrunsch/data"
    replicate_1_path = replicates_dir / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    image_type = "mip"
    square_size = 8000  # this indicates the size of image during prediction
    level = 0

    # Segmentation Model
    method = "cellpose"

    run_nucleus_features_extraction(replicate_1_path, image_type, level, square_size, method)
