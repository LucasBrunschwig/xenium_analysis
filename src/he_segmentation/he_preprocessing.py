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


def preprocess_he(img_: np.ndarray, square_size: int, separate_stain: bool = False):

    if separate_stain:
        # Extracted From QuPath
        stains = np.array([[0.651, 0.701, 0.29], [0.216, 0.801, 0.558]])
        h_and_e = separate_stains(img_, stains)
        hematoxylin = rescale_intensity(h_and_e[:, :, 0], out_range=(0, 255))
        eosin = rescale_intensity(h_and_e[:, :, 1], out_range=(0, 255))

    # Perform Stardist H&E
    model_type_ = "2D_versatile_he"
    print(img_.shape)
    model = StarDist2D.from_pretrained(model_type_)
    img_normalized = normalize(img_, 1, 99)

    prob_thrsh = None
    nms_thrsh = None

    print(img_.shape)
    print(f"Running Stardist: {model_type_}, h&e {square_size}")

    labels, details = model.predict_instances(img_normalized, prob_thresh=prob_thrsh, nms_thresh=nms_thrsh,)

    coord = details["coord"]

    print(f"Saving masks to: he_masks_stardist_{square_size}.pkl")

    with open(RESULTS / f"he_masks_stardist_{square_size}.pkl", "wb") as file:
        pickle.dump(file, coord)

    return coord


def load_he_masks(path_, model_version_, square_size_):
    raise ValueError("Not implemented")


if __name__ == "__main__":

    build_result_dir()

    human_breast_he_path = get_human_breast_he_path()

    ome_tiff = human_breast_he_path / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"

    image, metadata = load_xenium_he_ome_tiff(ome_tiff, level_=0)
    print(metadata)

    image, boundaries = image_patch(image, square_size_=8000)

    preprocess_he(image)



