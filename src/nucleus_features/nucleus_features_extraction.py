"""
Description: This python file has the purpose to extract most of the nuclei features. The goal is to use the
             segmentation information provided by different tools such as CellPose, Stardist, CellProfiler and
             compute different features to see how one  can classify nucleus based on nuclear morphology features.


Author: Lucas Brunschwig (lucas.brunschwig@epfl.ch)

Development:
[  ](1)Choose an algorithm and a type of image to extract the nucleus information
        - strategy 1: start with 2D clustering algorithm
        - strategy 2: build a good performing 3D algorithm

[  ](2) Find out how to extract nucleus individually for analysis
        - strategy 1: store masks.pkl (containing the label for each mask)
                      can be used to extract the pixel for each mask
        - remark: how to maintain spatial information of these mask.

Last Revision: 21.12.2023
"""

# Std
from pathlib import Path
from typing import Optional

# Third party
import numpy as np

# Relative import
from src.utils import load_image
from src.nucleus_segmentation.segmentation_watershed import segment_watershed
from src.nucleus_segmentation.segmentation_cellpose import segment_cellpose
from src.nucleus_segmentation.segmentation_stardist import segment_stardist
from src.nucleus_segmentation.utils import get_xenium_nucleus_boundaries, get_masks


SEGMENTATION = {"cellpose": segment_cellpose, "stardist": segment_stardist, "watershed": segment_watershed}


def load_masks(method_: str, dim_: str, path_replicate_: Path = Path(),
               square_size_: Optional[int] = None, img_type_: str = "mip"):
    """ This mehod will load masks """

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

    return get_masks(method, param, square_size_=square_size_, img_type_=img_type_)


def run_nucleus_features_extraction(path_replicate_: Path, image_type_: str, level_: int,
                                    image_dim_: str, method_: str):

    # Load Image
    img = load_image(path_replicate_, image_type_, level_)

    # Compute Masks
    masks = extract_masks(img, method_, image_dim_, path_replicate_)



    return 0


if __name__ == "__main__":

    # Image Parameters
    replicates_dir = Path("../../scratch/lbrunsch/data")
    replicate_1_path = replicates_dir / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    image_type = "mip"
    level = 0

    # Segmentation Model
    method = "xenium"
    image_dim = "2d"

    run_nucleus_features_extraction(replicate_1_path, image_type, level, image_dim, method)
