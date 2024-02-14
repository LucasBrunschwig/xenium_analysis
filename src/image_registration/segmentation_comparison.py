"""

The goal of this file is to create an accurate comparison between Stardist and H&E to measure the accuracy of stardist

Strategy:
- Register DAPI / H&E to get aligned nuclei
- Compute ground truth on 3 tiles (n x n) and then measure the accuracy of both models
    - the ground truth should represent DAPI and H&E signal -> create a good protocol for this
    - Protocol:
        - Open image in QuPath:
        - compute the background and stains on square: x = ,y = ,width = ,length =
        - Arrange the arrows such that it looks nice
        -
Remarks:
- how to superpose image while keeping the intensity separation

Implementations:
[ x ]: Extract stain vectors from QuPath
[ x ]: Superpose deconvolved hematoxylin H&E with DAPI in shades of gray
[   ]: Test different methods for superposition



"""
# Std
import os
import pickle
from pathlib import Path

# Third party
import cv2
import numpy as np
from tifffile import tifffile
import matplotlib.pyplot as plt

# Relative
import src.utils as src_utils
from src.nucleus_segmentation.segmentation_stardist import segment_stardist


def rgb_to_od(rgb, background):
    """Convert RGB to optical density (OD) space."""
    rgb = rgb.astype(np.float32) / background
    eps = 1e-8
    od_r = -np.log((rgb[:, :, 0] + eps) / (1 + eps))
    od_g = -np.log((rgb[:, :, 1] + eps) / (1 + eps))
    od_b = -np.log((rgb[:, :, 2] + eps) / (1 + eps))

    return od_r, od_g, od_b


def od_to_grayscale(od):
    """Convert optical density (OD) back to RGB space."""
    rgb = np.exp(-od) * 255
    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    return 255 - rgb.astype(int)


def color_deconvolution(image, stain_matrix, background):
    """Perform color deconvolution."""

    # Get the inverse of stain matrix
    stain_matrix_inv = np.linalg.inv(stain_matrix)

    # Convert RGB to OD
    od_r, od_g, od_b = rgb_to_od(image, background)

    # Deconvolve
    hematoxylin_r = od_r * stain_matrix_inv[0, 0]
    hematoxylin_g = od_g * stain_matrix_inv[1, 0]
    hematoxylin_b = od_b * stain_matrix_inv[2, 0]
    hematoxylin = hematoxylin_r + hematoxylin_b + hematoxylin_g

    return hematoxylin, None


def test_deconvolution(img, stains_method: str = "estimated", results_dir_: Path = Path()):

    # Define the stain vectors for H&E. These values are commonly used in literature.
    if stains_method == "standard":
        hematoxylin_stain = [0.65, 0.70, 0.29]
        eosin_stain = [0.07, 0.99, 0.11]
        residual_stain = [0.27, 0.57, 0.78]
        background = [255, 255, 255]

    elif stains_method == "estimated":
        hematoxylin_stain = [0.612, 0.735, 0.293]
        eosin_stain = [0.379, 0.895, 0.235]
        residual_stain = [-0.313, -0.115, 0.943]
        background = [236, 236, 237]
    else:
        raise ValueError("Not Implemented")

    # Normalize the stain vectors
    hematoxylin_stain /= np.linalg.norm(hematoxylin_stain)
    eosin_stain /= np.linalg.norm(eosin_stain)
    residual_stain /= np.linalg.norm(residual_stain)

    # Create the stain and background matrix
    stain_matrix = np.array([hematoxylin_stain, eosin_stain, residual_stain])
    background_vector = np.array(background)

    # Perform color deconvolution
    hematoxylin, eosin = color_deconvolution(img, stain_matrix, background_vector)
    # Remove the remaining

    thresholds = [-0.04, -0.02, 0.00, 0.02, 0.04]
    fig, axs = plt.subplots(ncols=len(thresholds)+1, nrows=1, figsize=(30, 5))
    [ax.axis("off") for ax in axs]

    # axs[0].imshow(od_to_grayscale(hematoxylin)[3000:3200, 1400:1600])
    # for i, threshold in enumerate(thresholds):
    #    hematoxylin_threshold = hematoxylin.copy()
    #    hematoxylin_threshold[hematoxylin_threshold < threshold] = threshold
    #    hematoxylin_threshold = od_to_grayscale(hematoxylin_threshold)
    #    axs[i+1].imshow(hematoxylin_threshold[3000:3200, 1400:1600])
    # plt.tight_layout()
    # fig.savefig("remove_background_thresholds.png")

    threshold = 0.07
    hematoxylin[hematoxylin < threshold] = threshold

    hematoxylin = od_to_grayscale(hematoxylin)

    # Save the results
    if results_dir_:
        tifffile.imwrite(str(results_dir_ / "hematoxylin_deconvolved_1024.tif"), hematoxylin)
        return hematoxylin, results_dir_ / "hematoxylin_deconvolved_1024.tif"
    return hematoxylin


def test_superposition(img_dapi, img_he, results_dir_):
    """

    :param img_dapi:
    :param img_he:
    :return:
    """
    if np.max(img_he) != np.max(img_dapi):
        img_dapi = cv2.normalize(img_dapi, None, 0, 255, cv2.NORM_MINMAX)

    center_ = [[img_dapi.shape[0] // 2 - 200, img_dapi.shape[0] // 2 + 200],
               [img_dapi.shape[1] // 2 - 200, img_dapi.shape[1] // 2 + 200]]

    img_dapi_float = img_dapi.astype(np.float16)
    img_he_float = img_he.astype(np.float16)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(5*3, 5*3))

    [axs[i, 0].imshow(img_dapi[center_[0][0]:center_[0][1], center_[1][0]:center_[1][1]]) for i in range(0, 2)]
    [axs[i, 1].imshow(img_he[center_[0][0]:center_[0][1], center_[1][0]:center_[1][1]]) for i in range(0, 2)]

    # Averaging
    averaged = (img_he_float + img_dapi_float) / 2

    axs[0, 2].imshow(averaged[center_[0][0]:center_[0][1], center_[1][0]:center_[1][1]])

    # Weighted
    weighted = img_he_float*0.3 + 0.7*img_dapi_float
    axs[1, 2].imshow(weighted[center_[0][0]:center_[0][1], center_[1][0]:center_[1][1]])

    plt.tight_layout()
    plt.savefig(results_dir_ / "superposition_methods.png")

    cv2.imwrite(str(results_dir_ / "superposed_averaged.png"), averaged.astype(int))
    cv2.imwrite(str(results_dir_ / "superposed_weighted.png"), weighted.astype(int))

    return averaged.astype(int), weighted.astype(int)


def test_stardist(img, model_type_, filename_, results_dir_):
    """ This function will run Stardist with same preprocessing as QuPath for comparison
        TODO: with ground truth compare the outputs

    :param img: input image
    :param filename_: the
    :param results_dir_:
    :return:
    """

    masks_dir = results_dir_ / "masks"
    os.makedirs(masks_dir, exist_ok=True)
    segmented = results_dir_ / "segmented"
    os.makedirs(segmented, exist_ok=True)
    original = results_dir_ / "original"
    os.makedirs(original, exist_ok=True)

    if len(img.shape) == 2:
        n_tiles = (1, 1)
    else:
        n_tiles = (1, 1, 1)

    masks_outline, _ = segment_stardist(img, model_type_=model_type_, n_tiles=n_tiles, do_3d=False, prob_thrsh=0.5,
                                        scale=0.2125/0.4)
    cv2.imwrite(str(original / filename_)+".png", img)
    with open(str(masks_dir / filename_) + ".pkl", "wb") as file:
        pickle.dump(masks_outline, file)

    if len(img.shape) == 2:
        img = cv2.merge([img, img, img])
    if np.max(img) > 256:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


    masks_color = [255, 0, 0]
    for mask in masks_outline:
        for coord in mask.T:
            if np.max(coord) < img.shape[0]-1:
                img[coord[0], coord[1]] = masks_color
                img[coord[0]+1, coord[1]] = masks_color
                img[coord[0]+1, coord[1]+1] = masks_color
                img[coord[0]+1, coord[1]-1] = masks_color
                img[coord[0]-1, coord[1]-1] = masks_color
                img[coord[0]-1, coord[1]] = masks_color
                img[coord[0], coord[1]-1] = masks_color

    cv2.imwrite(str(segmented / filename_)+"_segmented.png", img)

    return img


def test_segmentation(img, name, filenames, results_dir_):

    if len(img.shape) == 2:
        img = cv2.merge([img, img, img])

    masks_dir = results_dir_ / "masks"

    masks = []
    for filename in filenames:
        with open(str(masks_dir / filename) + ".pkl", 'rb') as file:
            masks.append(pickle.load(file))

    mask_color = [255, 0, 0]

    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(40, 7))
    [ax.axis("off") for ax in axs]

    for i, (masks_type, filename) in enumerate(zip(masks, filenames)):
        img_copy = img.copy()
        for mask in masks_type:
            for coord in mask.T:
                if np.max(coord) < img.shape[0] - 1:
                    img_copy[coord[0], coord[1]] = mask_color
                    img_copy[coord[0] + 1, coord[1]] = mask_color
                    img_copy[coord[0] + 1, coord[1] + 1] = mask_color
                    img_copy[coord[0] + 1, coord[1] - 1] = mask_color
                    img_copy[coord[0] - 1, coord[1] - 1] = mask_color
                    img_copy[coord[0] - 1, coord[1]] = mask_color
                    img_copy[coord[0], coord[1] - 1] = mask_color
        axs[i].imshow(img_copy)
        axs[i].set_title(filename)

    plt.savefig(results_dir_ / f"segmentation_comparison_{name}.png", dpi=500)


def build_results_dir():

    results_ = src_utils.get_results_path() / "comparison_DAPI_HE"
    os.makedirs(results_, exist_ok=True)

    return results_


if __name__ == "__main__":

    # ---------------------------------------- #
    # Script Parameters

    he_channel = "all"  # options: all, eosin, hematoxylin
    run_tests = True
    run_comparison = False

    # ---------------------------------------- #

    results_dir = build_results_dir()

    # Load H&E
    path_replicate = src_utils.get_human_breast_he_aligned_path()
    model_type_he = "2D_versatile_he"
    image_he = src_utils.load_xenium_he_ome_tiff(path_replicate, level_=0)

    # Load DAPI
    path_replicate = src_utils.get_human_breast_dapi_aligned_path()
    model_type_dapi = "2D_versatile_fluo"
    image_dapi = src_utils.load_xenium_he_ome_tiff(path_replicate, level_=0)

    if run_comparison:
        pass

    if run_tests:
        # Test the deconvolution of Hematoxylin on smaller image
        sqr = 512
        sub_img_he = image_he[image_he.shape[0] // 2 - sqr: image_he.shape[0] // 2 + sqr, image_he.shape[1] // 2 - sqr: image_he.shape[1] // 2 + sqr]
        sub_img_he_copy = sub_img_he.copy()
        sub_img_dapi = image_dapi[image_he.shape[0] // 2 - sqr: image_he.shape[0] // 2 + sqr, image_he.shape[1] // 2 - sqr: image_he.shape[1] // 2 + sqr]
        sub_img_dapi_norm = cv2.normalize(sub_img_dapi, None, 0, 255, cv2.NORM_MINMAX)

        sub_img_deconvolved, sub_img_path = test_deconvolution(sub_img_he, stains_method="estimated", results_dir_=results_dir)

        # Test the superposition of both images for
        image_averaged, image_weighted = test_superposition(sub_img_dapi, sub_img_deconvolved, results_dir_=results_dir)

        # Test Stardist
        img_he_seg = test_stardist(sub_img_he, model_type_="2D_versatile_he", filename_="H&E_1024", results_dir_=results_dir)
        img_he_dc_seg = test_stardist(sub_img_deconvolved, model_type_="2D_versatile_fluo", filename_="H&E_deconvolved", results_dir_=results_dir)
        img_dapi_seg = test_stardist(sub_img_dapi, model_type_="2D_versatile_fluo", filename_="DAPI_1024", results_dir_=results_dir)
        img_avg_seg = test_stardist(image_averaged, model_type_="2D_versatile_fluo", filename_="Averaged_1024", results_dir_=results_dir)
        img_wgt_seg = test_stardist(image_weighted, model_type_="2D_versatile_fluo", filename_="Weighted_1024", results_dir_=results_dir)

        fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(15, 3))
        [ax.axis("off") for ax in axs]
        axs[0].imshow(img_he_seg)
        axs[1].imshow(img_he_dc_seg)
        axs[2].imshow(img_dapi_seg)
        axs[3].imshow(img_avg_seg)
        axs[4].imshow(img_wgt_seg)

        plt.savefig(results_dir / "comparison_original.png")

        test_segmentation(sub_img_dapi_norm, name="dapi", filenames=["H&E_1024", "H&E_deconvolved", "DAPI_1024", "Averaged_1024", "Weighted_1024"], results_dir_=results_dir)
        test_segmentation(sub_img_he_copy, name="he", filenames=["H&E_1024", "H&E_deconvolved", "DAPI_1024", "Averaged_1024", "Weighted_1024"], results_dir_=results_dir)
