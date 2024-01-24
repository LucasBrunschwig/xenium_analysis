# Std Library
import os
import pickle
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import separate_stains
from skimage.exposure import rescale_intensity
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Relative Imports
from src.utils import load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, image_patch

RESULTS = Path()


def build_result_dir():
    global RESULTS
    RESULTS = get_results_path() / "he_preprocessing"
    os.makedirs(RESULTS, exist_ok=True)


def preprocess_he(img_: np.ndarray, square_size_: int, model_version_: str, separate_stain_: str = None,
                  prob_thrsh_: float = None, nms_thrsh_: float = None):

    # ------------------------------------------------------------------------------ #

    if separate_stain_:
        # Extracted From QuPath
        stains = np.array([[0.651, 0.701, 0.29], [0.216, 0.801, 0.558]])
        h_and_e = separate_stains(img_, stains)
        hematoxylin = rescale_intensity(h_and_e[:, :, 0], out_range=(0, 255))
        eosin = rescale_intensity(h_and_e[:, :, 1], out_range=(0, 255))

        if separate_stain_ == "hematoxylin":
            img_ = hematoxylin
        else:
            img_ = eosin

    # ------------------------------------------------------------------------------ #

    print(f"Running Stardist: {model_version_}, h&e {square_size}")

    # Perform Stardist H&E
    model = StarDist2D.from_pretrained(model_version_)
    img_normalized = normalize(img_, 1, 99)

    labels, details = model.predict_instances(img_normalized, prob_thresh=prob_thrsh_, nms_thresh=nms_thrsh_,
                                              n_tiles=(3, 3))

    coord = details["coord"]

    print(f"Saving masks to: he_masks_stardist_{square_size_}.pkl")

    with open(RESULTS / f"he_masks_stardist_{square_size_}.pkl", "wb") as file:
        pickle.dump(coord, file)

    return coord


def load_he_masks(path_, model_version_, square_size_):
    raise ValueError("Not implemented")


if __name__ == "__main__":

    # Scripts Parameters
    # ----------------------------------

    square_size = None  # The size of the image (from center)
    model_version = "2D_versatile_he"  # model from Stardist
    level = 0  # Pyramidal level: 0 = max resolution and 1 = min resolution
    separate_stains = None

    # ----------------------------------

    build_result_dir()

    # Load ome-tiff image
    human_breast_he_path = get_human_breast_he_path()
    ome_tiff = human_breast_he_path / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"

    # Use the custom loading for image and metadata
    image, metadata = load_xenium_he_ome_tiff(ome_tiff, level_=level)
    print(metadata)

    # Transform into a patch of the image
    image, boundaries = image_patch(image, square_size_=square_size)

    # Run Stardist Segmentation
    preprocess_he(image, square_size_=square_size, model_version_=model_version, separate_stain_=separate_stains)



