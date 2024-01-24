# Std Library
import os
import pickle
import platform
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import separate_stains
from skimage.exposure import rescale_intensity
from csbdeep.utils import normalize

# Relative Imports
from src.utils import load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, image_patch

# Temporary fix linked with Stardist issue in MacOS 14
if platform.platform() != "macOS-14.2.1-arm64-arm-64bit":
    from stardist.models import StarDist2D


RESULTS = Path()


def build_stardist_mask_outlines(masks):
    masks_outlines = []
    for mask in masks:
        mask = mask.astype(int)
        mask = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
        #tmp_1 = mask[0, :].copy()
        #tmp_2 = mask[1, :].copy()
        #mask[0, :] = tmp_2
        #mask[1, :] = tmp_1
        masks_outlines.append(mask)

    return masks_outlines


def build_result_dir():
    global RESULTS
    RESULTS = get_results_path() / "he_preprocessing"
    os.makedirs(RESULTS, exist_ok=True)


def preprocess_he(img_: np.ndarray, square_size_: int, model_version_: str, stain_: str = None,
                  prob_thrsh_: float = None, nms_thrsh_: float = None):

    # ------------------------------------------------------------------------------ #
    n_tiles = (3, 3, 1)
    if stain_:
        # Extracted From QuPath
        stains = np.array([[0.651, 0.701, 0.29], [0.216, 0.801, 0.558]])
        h_and_e = separate_stains(img_, stains)
        hematoxylin = rescale_intensity(h_and_e[:, :, 0], out_range=(0, 255))
        eosin = rescale_intensity(h_and_e[:, :, 1], out_range=(0, 255))

        if stain_ == "hematoxylin":
            img_ = hematoxylin
        else:
            img_ = eosin

        n_tiles = (3, 3)

    # ------------------------------------------------------------------------------ #

    print(f"Running Stardist: {model_version_}, h&e {square_size}")

    # Perform Stardist H&E
    model = StarDist2D.from_pretrained(model_version_)
    img_normalized = normalize(img_, 1,99.8, axis=(0, 1))

    labels, details = model.predict_instances(img_normalized, prob_thresh=prob_thrsh_, nms_thresh=nms_thrsh_,
                                              n_tiles=n_tiles)

    coord = details["coord"]

    results = f"he_masks_stardist_{square_size_}.pkl"
    if stain_:
        results = results[:-4] + stain_ + ".pkl"

    print(f"Saving masks to: {results}")

    with open(RESULTS / results, "wb") as file:
        pickle.dump(coord, file)

    return coord


def load_he_masks(path_: Path, model_version_, square_size_, visualize: bool = False, og_image: np.ndarray = None):

    with open(path_ / f"he_masks_stardist_{square_size_}.pkl", "rb") as file:
        masks_ = pickle.load(file)

    masks_ = build_stardist_mask_outlines(masks_)

    if visualize:
        if og_image is None:
            raise ValueError("Visualization expects the original image")
        viz_path = path_ / "viz"
        os.makedirs(viz_path, exist_ok=True)

        x_range = [image.shape[0] // 2 - 200, image.shape[0] // 2 + 200]
        y_range = [image.shape[1] // 2 - 200, image.shape[1] // 2 + 200]

        plt.figure()
        plt.imshow(image[x_range[0]:x_range[1], y_range[0]:y_range[1]])
        for mask in masks_:
            if check_ranges(mask, x_range, y_range):
                x = mask[0, :] - x_range[0]
                y = mask[1, :] - y_range[0]
                plt.plot(x, y, 'r', linewidth=.8)
        plt.savefig(viz_path / f"he_masks_stardist_{square_size_}.png")

    return masks_


def check_ranges(mask, x_range, y_range):
    return ((x_range[0] < mask[0, :].max() < x_range[1] and x_range[0] < mask[0, :].min() < x_range[1]) and
            (y_range[0] < mask[1, :].max() < y_range[1] and y_range[0] < mask[1, :].min() < y_range[1]))


if __name__ == "__main__":

    # Scripts Parameters
    # ----------------------------------

    square_size = None  # The size of the image (from center)
    model_version = "2D_versatile_he"  # model from Stardist
    level = 0  # Pyramidal level: 0 = max resolution and 1 = min resolution
    stains = None
    run_stardist = True  # run stardist or load masks

    # ----------------------------------

    build_result_dir()

    # Load ome-tiff image
    human_breast_he_path = get_human_breast_he_path()
    ome_tiff = human_breast_he_path / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"

    # Use the custom loading for image and metadata
    image, metadata = load_xenium_he_ome_tiff(ome_tiff, level_=level)
    print(metadata)

    # Transform into a patch of the image
    image, boundaries = image_patch(image, square_size_=square_size, type_="HE")

    # Run Stardist Segmentation or load masks
    if run_stardist:
        masks = preprocess_he(image, square_size_=square_size, model_version_=model_version, stain_=stains)
    else:
        masks = load_he_masks(RESULTS, model_version, square_size, visualize=True, og_image=image)


