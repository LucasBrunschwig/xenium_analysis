"""
Description: The goal of this script is to compare the predictions of stardist on H&E and DAPI at the same location.
             The trick is that this requires an alignment between DAPI and H&E which is performed by he_alignment.py
             This implies that we require different resolution between DAPI and H&E because of the different tissue
             section.

             One of the goal is to see if we could potentially use the masks predict on H&E as a ground truth to fine
             tune stardist for DAPI images.

Thoughts: We could simply upscale/downscale masks to compare masks on different resolutions for both of them.
          This is why, we need to determine the impact of resolution on the image. Perhaps fine-tuning the highest
          resolution will be better in the end.

          We have both masks on the highest resolution. For H&E the masks is on the full image but we have the formula
          to transform to aligned coordinates. For DAPI we have stardist predictions on best resolution. We could
          divide by two each of them to get the corresponding masks.

          It would still be interesting to see how DAPI perform on different resolution for this dataset.


Procedure:
[ x ]: load both images on predefined resolution and aligned
[ x ]: visualize MIP  and H&E as a first comparison
[   ]: visualize masks from H&E on H&E aligned and DAPI
[ ]: decide if we are going to use prediction on the original H&E + transformation or the aligned H&E (lower res)
[ ]: Check same locations for Stardist and show both predictions

Parameters:
    - dapi resolution
    - location to compare

"""

# Std Library
import os
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

# Relative Imports
from src.utils import get_human_breast_he_path, get_results_path, load_image, load_xenium_he_ome_tiff, load_xenium_data
from src.nucleus_segmentation.segmentation_stardist import build_stardist_mask_outlines
from src.nucleus_segmentation.utils import get_xenium_nucleus_boundaries

def build_results_dir():
    results_dir_ = get_results_path() / "he_comparison"
    os.makedirs(results_dir_, exist_ok=True)

    return results_dir_


def check_ranges(mask, x_range, y_range):
    return ((x_range[0] < mask[0, :].max() < x_range[1] and x_range[0] < mask[0, :].min() < x_range[1]) and
            (y_range[0] < mask[1, :].max() < y_range[1] and y_range[0] < mask[1, :].min() < y_range[1]))


def conversion(masks, crop_, scale_, affine_, img_he_):
    """ The current masks were predicted on the whole image """
    new_masks = []

    img_he_flip_ = cv2.flip(img_he, 0)

    for mask in masks:
        new_mask = []
        for coord in mask.T:

            # vertical flip

            coord[0] = 24241 - coord[0]  # validated with one nucleus

            # DEBUG
            # plt.imshow(img_he_[mask[0].min():mask[0].max(),mask[1].min():mask[1].max()])
            # plt.plot(mask[1] - mask[1].min(), mask[0]-mask[0].min())
            # mask[0,:] [el[0] - 24241 for el in mask.T]

            position_scaled = [int(el * scale_) for el in coord]

            position_cropped = position_scaled.copy()

            position_cropped[0] = position_scaled[0] - crop_[1]
            position_cropped[1] = position_scaled[1] - crop_[0]

            coord_aligned = [int(el) for el in np.matmul(affine_, np.array([position_cropped[1], position_cropped[0], 1]))]

            new_mask.append([coord_aligned[1], coord_aligned[0]])

        if new_mask:
            new_masks.append(np.array(new_mask).T)

    return new_masks


def comparison(positions_, square_size_, img_he_, img_dapi_, masks_, xenium_masks_, save_dir_):
    fig, axs = plt.subplots(nrows=3, ncols=len(positions_), figsize=(5 * len(positions_), 10))
    [ax.axis("off") for ax in axs.ravel()]

    for i, pos in enumerate(positions_):
        x_range = [pos[0] - square_size_, pos[0] + square_size_]
        y_range = [pos[1] - square_size_, pos[1] + square_size_]

        axs[0, i].imshow(
            img_he_[pos[0] - square_size_:pos[0] + square_size_, pos[1] - square_size_:pos[1] + square_size_])
        axs[1, i].imshow(
            img_dapi_[pos[0] - square_size_:pos[0] + square_size_, pos[1] - square_size_:pos[1] + square_size_])

        for mask in masks_:
            if check_ranges(mask, x_range=x_range, y_range=y_range):
                x = mask[0, :] - x_range[0]
                y = mask[1, :] - y_range[0]
                axs[0, i].plot(y, x, 'r', linewidth=.8)
                axs[1, i].plot(y, x, 'r', linewidth=.8)

        # Plot the Xenium Dataset
        axs[2, i].imshow(img_dapi_[pos[0] - square_size_:pos[0] + square_size_, pos[1] - square_size_:pos[1] + square_size_])

        for mask in xenium_masks_:
            if check_ranges(mask, x_range=y_range, y_range=x_range):
                x = mask[0, :] - y_range[0]
                y = mask[1, :] - x_range[0]
                axs[2, i].plot(x, y, 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(save_dir_ / f"he_dapi_comparison_{square_size_}.png")


def visualize(positions_, square_size_, img_he_, img_dapi_, save_dir_):

    fig, axs = plt.subplots(nrows=2, ncols=len(positions_), figsize=(5*len(positions_), 10))
    [ax.axis("off") for ax in axs.ravel()]

    for i, pos in enumerate(positions_):
        axs[0, i].imshow(img_he_[pos[0]-square_size_:pos[0]+square_size_, pos[1]-square_size_:pos[1]+square_size_])
        axs[1, i].imshow(img_dapi_[pos[0]-square_size_:pos[0]+square_size_, pos[1]-square_size_:pos[1]+square_size_])

    plt.tight_layout()
    plt.savefig(save_dir_ / f"he_dapi_comparison_{square_size_}.png")
    plt.close()


if __name__ == "__main__":

    # Scripts Parameters
    # ----------------------------------

    dapi_level = 1  # defined by the resolution on which H&E was aligned
    img_type = "mip"  # defined by the type of image on which H&E was aligned
    locations = [(9609, 4388), (7882, 4733), (7711, 3651), (8467, 10561), (2286, 9441)]  # the location where image will be compared
    square_size = [300, 150, 50]

    # ----------------------------------

    results_dir = build_results_dir()

    img_he_aligned = cv2.imread(str(get_results_path() / "he_alignment" / "Human_Breast_Replicate1_HE_aligned_V2.tif"))
    img_he_aligned = cv2.imread("/Users/lbrunsch/Desktop/HE_hematoxylin_gray_aligned.png")

    img_dapi = load_image(get_human_breast_he_path(), level_=dapi_level, img_type=img_type)

    xenium_boundaries = get_xenium_nucleus_boundaries(get_human_breast_he_path(), scale=0.5, save=True)

    # In matplotlib first coordinate is vertical, second coordinate is horizontal

    print("# --------------------------------------- #")
    print("1. Visualize DAPI and H&E at different locations side by side")

    for size_ in square_size:
        visualize(positions_=locations, square_size_=size_, img_he_=img_he_aligned, img_dapi_=img_dapi,
                  save_dir_=Path("/Users/lbrunsch/Desktop"))

    print("# --------------------------------------- #")
    print("2. Masks Conversion from original image to new coordinates")

    scale_factor = 0.363788 / 0.425

    # Matrix for V2 aligned
    transform_matrix = np.array([[1.000572656367927, -0.005233918480924, 31.872460487932152],
                                 [0.005143485132005, 1.000008388932138, -24.059854887329493]])

    transform_matrix_hematoxylin_gray = np.array([[1.000503593904112, -0.004533845798186, 27.518744606547656],
                                                  [0.004928549748289, 1.000133333813246, -15.419156094757454]])
    crop = [5137, 2101]

    crop = [5135, 2109]

    with open(get_results_path() / "he_preprocessing" / f"he_masks_stardist_None.pkl", "rb") as file:
        masks = pickle.load(file)

    masks = build_stardist_mask_outlines(masks)

    # Load ome-tiff images
    human_breast_path = get_human_breast_he_path()
    he_ome_tiff = human_breast_path / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"

    # Use custom loading for H&E image and metadata
    print("Loading H&E...")
    img_he, metadata = load_xenium_he_ome_tiff(he_ome_tiff, level_=0)
    print("metadata", metadata)

    masks = conversion(masks, crop, scale_factor, transform_matrix_hematoxylin_gray, img_he)

    comparison(positions_=locations, square_size_=50, img_he_=img_he_aligned, img_dapi_=img_dapi, masks_=masks,
               xenium_masks_=xenium_boundaries, save_dir_=Path("/Users/lbrunsch/Desktop"))


