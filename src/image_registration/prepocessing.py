"""
In this image we will preprocess both images and align them with one another


Strategy:
- invert DAPI so that it matches H&E nuclei in terms of correlation and convert H&E to shades of gray
- noise reduction with Gaussian smoothing
    - applying gaussian noise to both images before resizing can help in getting similar amount of details
- Use the highest resolution possible of h&e
- resizing h&e to upper resolution because difference in resolution are minor.
    - the choice is to maintain the DAPI images because we are interested in analyzing the inside of nuclei.
    - the change of size for h&e does not impact too much since we are only considering the border.
    - however, it is crucial to investigate the alignment of h&e nuclei with DAPI.
- test different methods:
    - feature-based: ran nuclei segmentation on both images roughly align images with any method and select nuclei
      that overlap with more than 0.8
    - intensity-based: different algorithm
    - choose between linear and non-linear alignments
- cropping: template matching and cropping
- quality check:
    - use nuclei segmentation from before registration, follow the same alignment as for registration and observe
      how they superimpose with nuclei
    - something that could be interesting would be to quantify the alignment with some grounds truth in DAPI only in TP
      to see how well they overlap.

Implementations:
[ x ]: Test different methods for up-scaling and down-scaling and choose one
       -> visually up-scaling seems to not introduce significant artifact
       -> selected method for interpolation = cv2.INTER_CUBIC
[ x ]: Test gaussian smoothing with various sigma before or after upscaling
       -> compute sigma based on estimated nuclei size since this is what we would like to segment
       ->
[   ]: for feature based alignment:
       - run stardist and

"""
import pickle
from typing import Optional
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac
from src.nucleus_segmentation.segmentation_stardist import build_stardist_mask_outlines
#from wsireg.wsireg2d import WsiReg2D

from src.utils import load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, load_image

# -------------------------------------- #
# HELPER METHODS


def resize(img_, scale_factor_, interpolation: str = "cv2.INTERCUBIC"):
    """ Resize an image based on scale factor and interpolation method """

    interpolation = eval(interpolation)
    new_dim = (int(img_.shape[1]*scale_factor_), int(img_.shape[0]*scale_factor_))
    return cv2.resize(img_, new_dim, interpolation=interpolation)


def template_match(img_dapi, img_he, matching, padding: int = 1000, convert_gray: bool = True):
    matching = eval(matching)

    # Convert H&E to shades of gray
    if convert_gray:
        img_he_gray = cv2.cvtColor(img_he, cv2.COLOR_BGR2GRAY)
    else:
        img_he_gray = img_he

    # Scale dapi and invert
    img_dapi_scaled = cv2.normalize(img_dapi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img_dapi_invert = 255 - img_dapi_scaled

    # Template matching
    result = cv2.matchTemplate(img_he_gray, img_dapi_invert, matching)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    height, width = img_dapi_invert.shape[:2]

    # Return matches template
    img_he_aligned = crop_image(img_he, top_left[1]-padding, top_left[0]-padding,
                                width+2*padding, height+2*padding)

    return img_he_aligned


def crop_image(img, x_, y_, width, height):
    """ x is the vertical coordinate, y is the horizontal coordinate"""
    return img[x_:x_ + height, y_:y_ + width]


def flip_image(img):
    return cv2.flip(img, 0)


def gaussian_blurring(img, sigma):
    return gaussian_filter(img, sigma, mode="reflect")


def segment_stardist(
        img: np.ndarray,
        model_type_: str,
        prob_thrsh: Optional[float],
        nms_thrsh: Optional[float],
        n_tiles: tuple,
):
    model = StarDist2D.from_pretrained(model_type_)

    img_normalized = normalize(img, 1, 99.8, axis=(0, 1))
    labels, details = model.predict_instances(img_normalized, prob_thresh=prob_thrsh, nms_thresh=nms_thrsh,
                                              n_tiles=n_tiles)

    return labels, details

# -------------------------------------- #
# TEST METHODS


def test_resize(img_, dapi_res_0: float = 0.2125, dapi_res_1: float = 0.425, he_res: float = 0.3637,
                results_dir_: Path = Path()):

    test_resize_dir = results_dir_ / "test_resize"
    os.makedirs(test_resize_dir, exist_ok=True)

    dapi_resolutions = [dapi_res_0, dapi_res_1]
    up_sampling_methods = ["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_LANCZOS4"]
    down_sampling_methods = ["cv2.INTER_AREA", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_LANCZOS4"]
    sampling_methods = [up_sampling_methods, down_sampling_methods]
    names = ["up", "down"]

    locations = [[img_.shape[0] // 2, img_.shape[1] // 2], [9550, 7320], [10470, 17540]]

    for sampling_method, dapi_res, name in zip(sampling_methods, dapi_resolutions, names):

        print(f"Testing: {name}-sampling methods on H&E level 0 ({he_res} -> {dapi_res})")

        fig, axs = plt.subplots(nrows=len(locations), ncols=len(sampling_method)+1, figsize=(30, 15))
        [ax.axis("off") for ax in axs.ravel()]

        for j, loc_ in enumerate(locations):

            scale_factor = he_res / dapi_res

            sqr = 15
            axs[j, 0].imshow(img_[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
            axs[j, 0].set_title("Original")

            sqr = int(sqr * scale_factor)
            loc_ = [int(loc_[0] * scale_factor), int(loc_[1] * scale_factor)]

            for i, method_ in enumerate(sampling_method):

                img_he_resized = resize(img_, scale_factor, method_)

                # Visualize Up-sampling
                axs[j, i+1].imshow(img_he_resized[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
                axs[j, i+1].set_title(method_)

        plt.tight_layout()
        plt.savefig(test_resize_dir / f"{name}_sampling_comparison.png")
        plt.close(fig)


def test_noise_reduction(img_he_, results_dir):
    """
    The goal is to visualize the impact of gaussian smoothing on the image if performed before or after resizing

    Nuclei are of size 6-10 microns
        - we have 0.36 microns/pixel <-> one nucleus is between 15-30 pixels
        - after upscaling: 0.2125 microns/pixel <-> one nucleus 30-50 pixels

    For gaussian smoothing, sigma is exerted on int(sigma*4) pixels:

    Resolution: 0.36
        - we want to impact 1/5 of the nuclei (between 3, 6 pixels)
        - sigma: 0.5 -> 2 pixels, sigma: 2 -> 8 pixels
        - Hence, we will test original resolution on sigma = [0.5, 1, 2, 3]
    Resolution: 0.2125
        - we want to impact 1/5 of the nuclei (between 6, 10 pixels)
        - sigma: 1 -> 4 pixels, sigma 3 -> 12 pixels
        - Hence, we will test on sigma = [1, 2, 3, 4]

    """

    img_he_ = cv2.cvtColor(img_he_, cv2.COLOR_BGR2GRAY)

    gaussian_noise_dir = results_dir / "test_gaussian_noise"
    os.makedirs(gaussian_noise_dir, exist_ok=True)

    # Test Gaussian Blur Before Up-Sampling

    sigmas = [0.5, 1, 2, 3]
    method = "cv2.INTER_CUBIC"
    sqr = 20
    scale_factor = 0.3637 / 0.2125
    sqr_resized = int(sqr * scale_factor)

    locations = [[img_he_.shape[0] // 2, img_he_.shape[1] // 2], [9550, 7320], [10470, 17540]]
    locations_resized = [[int(loc_[0] * scale_factor), int(loc_[1] * scale_factor)] for loc_ in locations]

    fig, axs = plt.subplots(nrows=len(locations), ncols=len(sigmas)+1, figsize=(30, 15))
    fig_resized, axs_resized = plt.subplots(nrows=len(locations), ncols=len(sigmas)+1, figsize=(30, 15))

    [ax.axis("off") for ax in axs.ravel()]
    [ax.axis("off") for ax in axs_resized.ravel()]

    axs[0, 0].set_title("Original")
    axs_resized[0, 0].set_title("Original")

    for i, sigma in enumerate(sigmas):
        img_he_filtered = gaussian_blurring(img_he_, sigma)
        img_he_filtered_resized = resize(img_he_filtered, scale_factor_=scale_factor, interpolation=method)
        axs[0, i+1].set_title(f"sigma {sigma}")
        axs_resized[0, i+1].set_title(f"sigma {sigma}")
        for j, loc_ in enumerate(locations):
            axs[j, i+1].imshow(img_he_filtered[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
        for j, loc_ in enumerate(locations_resized):
            axs_resized[j, i+1].imshow(img_he_filtered_resized[loc_[0]-sqr_resized:loc_[0]+sqr_resized,
                                                               loc_[1]-sqr_resized:loc_[1]+sqr_resized])

    for j, loc_ in enumerate(locations):
        axs[j, 0].imshow(img_he_[loc_[0] - sqr:loc_[0] + sqr, loc_[1] - sqr:loc_[1] + sqr])

    img_he_resized = resize(img_he_, scale_factor, method)
    for j, loc_ in enumerate(locations_resized):
        axs_resized[j, 0].imshow(img_he_resized[loc_[0] - sqr_resized:loc_[0] + sqr_resized,
                                                loc_[1] - sqr_resized:loc_[1] + sqr_resized])

    plt.tight_layout()
    fig.savefig(gaussian_noise_dir / "original_image_gaussian_smoothing.png")
    plt.close(fig)
    plt.tight_layout()
    fig_resized.savefig(gaussian_noise_dir / "original_image_gaussian_smoothing_upscaled.png")
    plt.close(fig_resized)

    # Test Gaussian Blur after Up-Sampling

    sigmas = [1, 2, 3]

    fig, axs = plt.subplots(nrows=len(locations), ncols=len(sigmas)+1, figsize=(30, 15))
    [ax.axis("off") for ax in axs.ravel()]

    for j, loc_ in enumerate(locations_resized):
        axs[j, 0].imshow(img_he_resized[loc_[0] - sqr_resized:loc_[0] + sqr_resized,
                                        loc_[1] - sqr_resized:loc_[1] + sqr_resized])

    axs[0, 0].set_title("Original")
    for i, sigma in enumerate(sigmas):
        img_he_filtered = gaussian_blurring(img_he_resized, sigma)
        axs[0, i+1].set_title(f"sigma {sigma}")
        for j, loc_ in enumerate(locations_resized):
            axs[j, i+1].imshow(img_he_filtered[loc_[0]-sqr_resized:loc_[0]+sqr_resized,
                                               loc_[1]-sqr_resized:loc_[1]+sqr_resized])

    plt.tight_layout()
    fig.savefig(gaussian_noise_dir / "upsampled_image_gaussian_smoothing.png")
    plt.close(fig)


def test_stardist_features_alignment(img_he_: np.ndarray, img_dapi_: np.ndarray, results_dir_: Path):
    """
        1. Run Stardist on both images
        2. Run an image based alignments
        3. Compute centroid of masks and select masks that are less than x pixels based
        4. Run feature based alignment with nuclei centroid
        5. need to check how well this masks are aligned with the original cells

        Questions:
            - how many landmarks should we use? Should I visualize all of them?
            - should I consider the masks resolution here
    """

    save_masks = results_dir_ / "masks"
    os.makedirs(save_masks, exist_ok=True)

    mask_path = save_masks / "results_stardist_dapi_0_nms-None_thrsh-None.pkl"

    if not os.path.isfile(mask_path):
        model_type_fluo = "2D_versatile_fluo"
        labels_dapi, masks_dapi = segment_stardist(img_dapi_, model_type_fluo, prob_thrsh=None, nms_thrsh=None,
                                                   n_tiles=(10, 10))
        with open(mask_path, 'wb') as file:
            pickle.dump(masks_dapi, file)
    else:
        with open(mask_path, 'rb') as file:
            masks_dapi = pickle.load(file)

    mask_path = save_masks / "results_stardist_he_nms-None_thrsh-None.pkl"

    if not os.path.isfile(mask_path):
        model_type_he = "2D_versatile_he"
        labels_he, masks_he = segment_stardist(img_he_, model_type_he, prob_thrsh=None, nms_thrsh=None,
                                               n_tiles=(10, 10, 1))
        with open(mask_path, 'wb') as file:
            pickle.dump(masks_he, file)

    else:
        with open(mask_path, 'rb') as file:
            masks_dapi = pickle.load(file)

    # Here I need to compute distances between pairs of point take the top x and print the region
    # Probably needs to align image before with a rough estimate -> more complicated than anticipated


def test_anisotropic_filter(img_he_, results_dir_):

    img_he_ = cv2.cvtColor(img_he_, cv2.COLOR_BGR2GRAY)

    gaussian_noise_dir = results_dir_ / "test_anisotropic_filter"
    os.makedirs(gaussian_noise_dir, exist_ok=True)

    # Test Gaussian Blur Before Up-Sampling

    weights = [0.1, 0.3, 0.5]
    method = "cv2.INTER_CUBIC"
    sqr = 20
    scale_factor = 0.3637 / 0.2125
    sqr_resized = int(sqr * scale_factor)

    locations = [[img_he_.shape[0] // 2, img_he_.shape[1] // 2], [9550, 7320], [10470, 17540]]
    locations_resized = [[int(loc_[0] * scale_factor), int(loc_[1] * scale_factor)] for loc_ in locations]

    fig, axs = plt.subplots(nrows=len(locations), ncols=len(weights)+1, figsize=(30, 15))
    fig_resized, axs_resized = plt.subplots(nrows=len(locations), ncols=len(weights)+1, figsize=(30, 15))

    [ax.axis("off") for ax in axs.ravel()]
    [ax.axis("off") for ax in axs_resized.ravel()]

    axs[0, 0].set_title("Original")
    axs_resized[0, 0].set_title("Original")

    for i, weight in enumerate(weights):
        img_he_filtered = denoise_tv_chambolle(img_he_, weight)
        img_he_filtered_resized = resize(img_he_filtered, scale_factor_=scale_factor, interpolation=method)
        axs[0, i+1].set_title(f"iteration {weight}")
        axs_resized[0, i+1].set_title(f"iteration {weight}")
        for j, loc_ in enumerate(locations):
            axs[j, i+1].imshow(img_he_filtered[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
        for j, loc_ in enumerate(locations_resized):
            axs_resized[j, i+1].imshow(img_he_filtered_resized[loc_[0]-sqr_resized:loc_[0]+sqr_resized,
                                                               loc_[1]-sqr_resized:loc_[1]+sqr_resized])

    for j, loc_ in enumerate(locations):
        axs[j, 0].imshow(img_he_[loc_[0] - sqr:loc_[0] + sqr, loc_[1] - sqr:loc_[1] + sqr])

    img_he_resized = resize(img_he_, scale_factor, method)
    for j, loc_ in enumerate(locations_resized):
        axs_resized[j, 0].imshow(img_he_resized[loc_[0] - sqr_resized:loc_[0] + sqr_resized,
                                                loc_[1] - sqr_resized:loc_[1] + sqr_resized])

    plt.tight_layout()
    fig.savefig(gaussian_noise_dir / "original_image_anisotropic.png")
    plt.close(fig)
    plt.tight_layout()
    fig_resized.savefig(gaussian_noise_dir / "original_anisotropic_upscaled.png")
    plt.close(fig_resized)


def test_feature_based_registration(feature_image, search_image):

    # Detect and extract features
    orb = ORB(n_keypoints=500, fast_threshold=0.05)

    orb.detect_and_extract(feature_image)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    orb.detect_and_extract(search_image)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    # Match features
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Select matched keypoints
    src = keypoints2[matches[:, 1]][:, ::-1]
    dst = keypoints1[matches[:, 0]][:, ::-1]

    # Estimate the transformation model using RANSAC for robustness
    model, inliers = ransac((src, dst), AffineTransform, min_samples=4,
                            residual_threshold=2, max_trials=1000)

    # Warp the moving image towards the fixed image
    registered_image = warp(search_image, inverse_map=model.inverse, output_shape=feature_image.shape[:2])

    return registered_image, model


def test_wsireg(img_dapi_path_, img_he_path_, transform_list, results_dir_):

    transform_ = "wsireg_" + "_".join(transform_list)
    wsireg_results = results_dir_ / transform_
    os.makedirs(wsireg_results, exist_ok=True)

    reg_graph = WsiReg2D("registration", wsireg_results)

    reg_graph.add_modality(
        "DAPI",
        str(img_dapi_path_),
        0.2125,
        channel_names=["DAPI"],
        preprocessing={
            "image_type": "FL",
            "as_uint8": True,
            "ch_indices": None,
            "contrast_enhance": False,
        }
    )

    reg_graph.add_modality(
        "HE",
        str(img_he_path_),
        0.2125,
        preprocessing={
            "image_type": "BF",
            "as_uint8": True,
        },
        channel_names=["R", "G", "B", ],
        channel_colors=["red", "green", "blue"],
    ) # BF invert intensity

    reg_graph.add_reg_path(
        "DAPI",
        "HE",
        thru_modality=None,
        reg_params=transform_list
    )

    reg_graph.register_images()

    reg_graph.save_transformations()

    reg_graph.transform_images(file_writer="ome.tiff")

    # Aligned files
    output_file = wsireg_results / "registration-DAPI_to_HE_registered.ome.tiff"
    img_he_registered = tifffile.imread(str(output_file))
    img_he_registered_aligned = template_match(img_dapi, img_he_registered, matching="cv2.TM_CCOEFF", padding=0,
                                               convert_gray=True)
    img_registered_name = "HE_registered.tif"
    tifffile.imwrite(wsireg_results / img_registered_name, img_he_registered_aligned)


def test_wsireg_alignments(img_he_registered_, img_dapi_, results_dir_, he_registration_type):

    # Create directories
    wsireg_alignment_path = results_dir_ / "wsireg_alignments"
    os.makedirs(wsireg_alignment_path, exist_ok=True)

    # Store Masks for each registration
    mask_dir = wsireg_alignment_path / "masks"
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = mask_dir / f"masks_{he_registration_type}"

    # Create masks for He registered
    if not os.path.isfile(mask_path):
        model_type_he = "2D_versatile_he"
        labels_he, masks_he = segment_stardist(img_he_registered_, model_type_he, prob_thrsh=None, nms_thrsh=None,
                                               n_tiles=(5, 5, 1))
        with open(mask_path, 'wb') as file:
            pickle.dump(masks_he, file)

    else:
        with open(mask_path, 'rb') as file:
            masks_he = pickle.load(file)

    masks_he = build_stardist_mask_outlines(masks_he["coord"])

    # Load DAPI Masks
    dapi_mask_path = results_dir_ / "masks" / "results_stardist_dapi_0_nms-None_thrsh-None.pkl"
    with open(dapi_mask_path, "rb") as file:
        masks_dapi = pickle.load(file)

    masks_dapi = build_stardist_mask_outlines(masks_dapi["coord"])

    locations = [[img_he_registered_.shape[0] // 2, img_he_registered_.shape[1] // 2], [9550, 7320], [10470, 17540],
                 [1000, 1000], [img_he_registered_.shape[0]-500, img_he_registered[1].shape[1]-500]]

    fig, axs = plt.subplots(nrows=3, ncols=len(locations), figsize=(20, 10))
    [ax.axis("off") for ax in axs.ravel()]

    for i, loc_ in enumerate(locations):
        range_ = 200
        x_range = [loc_[0]-range_, loc_[0]+range_]
        y_range = [loc_[1]-range_, loc_[1]+range_]
        axs[0, i].imshow(img_he_registered_[x_range[0]: x_range[1], y_range[0]: y_range[1]])
        axs[1, i].imshow(img_dapi_[x_range[0]: x_range[1], y_range[0]: y_range[1]])

        for mask in masks_he:
            if check_ranges(mask, x_range, y_range):
                x = mask[0, :] - x_range[0]
                y = mask[1, :] - y_range[0]
                axs[0, i].plot(y, x, 'r', linewidth=.8)
                axs[1, i].plot(y, x, 'r', linewidth=.8)

        axs[2, i].imshow(img_dapi_[x_range[0]: x_range[1], y_range[0]: y_range[1]])
        for mask in masks_dapi:
            if check_ranges(mask, x_range, y_range):
                x = mask[0, :] - x_range[0]
                y = mask[1, :] - y_range[0]
                axs[2, i].plot(y, x, 'r', linewidth=.8)
    plt.savefig(wsireg_alignment_path / f"alignment_{wsireg_type}.png")


def check_ranges(mask, x_range, y_range):
    return ((x_range[0] < mask[0, :].max() < x_range[1] and x_range[0] < mask[0, :].min() < x_range[1]) and
            (y_range[0] < mask[1, :].max() < y_range[1] and y_range[0] < mask[1, :].min() < y_range[1]))


# -------------------------------------- #
# MAIN METHODS


def preprocess_he_image(img, x_: int, y_: int, w_: int, h_: int, save_img: Optional[Path] = None):
    """
    Descr: The original h&e image contains the xenium panels and is vertically flipped compared to DAPI this method
           flip and crop the image based on value visually determined.

    -  1. Crop image with a bounding box (x, y, w, h)
    -  2. flip image and save it under a new format
    """

    img = crop_image(img, x_=x_, y_=y_, width=w_, height=h_)
    img = flip_image(img)

    if save_img is not None:
        tifffile.imwrite(save_img / "Xenium_FFPE_Human_Breast_Cancer_Preprocessed.tif", img)

    # Gaussian Smoothing and Resizing

    return img


def alignment_feature_based(img_he_, img_dapi_):
    """
    Description: the goal here is to test feature alignment. However, we need to identify these features
        1. Run Stardist on both images
        2. Do a rough alignments and find nuclei that are close to one another


    """
    pass


def alignment_intensity_based(img_he_, img_dapi_):
    pass


def build_results_dir():
    results_path = get_results_path()
    results_path = results_path / "image_registration"
    os.makedirs(results_path, exist_ok=True)
    return results_path


if __name__ == "__main__":

    # ---------------------------------------- #
    # Script Parameters

    he_channel = "all"  # options: all, eosin, hematoxylin
    dapi_level = 0  # dapi resolution: 2*0.2125 = 0.425 microns / pixel
    img_type = "mip"
    he_level = 0  # resolution: 0.3638 microns / pixel
    run_tests = True
    run_alignment = False

    # ---------------------------------------- #
    # General Predefined parameters
    hb_path = get_human_breast_he_path()
    results_dir = build_results_dir()

    # ---------------------------------------- #
    # Main Methods

    img_he_path = get_human_breast_he_path() / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"
    img_he, metadata = load_xenium_he_ome_tiff(img_he_path, level_=he_level)
    img_dapi_path = get_human_breast_he_path() / "morphology_mip.ome.tif"
    img_dapi = load_image(get_human_breast_he_path(), level_=dapi_level, img_type=img_type)

    tifffile.imwrite(results_dir / "Xenium_DAPI_Level_0.tif", img_dapi)

    #x, y, h, w = 3680, 3795, 18977, 25150
    x, y, h, w = 5680, 3795, 16977, 25150

    # Flip and Remove Xenium Panels from image
    img_he = preprocess_he_image(img_he, x, y, w, h, results_dir)

    if run_alignment:
        pass
        # Step 1: Ensure that both images have the same resolution

        # Step 2: Apply template matching and add padding such that both images are not too far

        # Step 3: Apply noise removal

        # Step 4: Invert DAPI image to have similar

        # Step 5: Perform the alignment

    # ---------------------------------------- #

    if run_tests:

        # Test Up-sampling vs Down-sampling
        # test_resize(img_he, results_dir_=results_dir)

        # Test Different Noise Reduction + before vs after up-sampling
        # test_noise_reduction(img_he, img_dapi_=None, results_dir=results_dir)
        # test_anisotropic_filter(img_he, results_dir)

        # Run Stardist on original image
        # test_stardist_features_alignment(img_he_=img_he, img_dapi_=img_dapi, results_dir_=results_dir)

       # This test requires the image to be up-scaled - matched+padding with the original tissue

        transforms = [["rigid"], ["rigid", "affine"], ["rigid", "affine", "nl"], ["affine", "nl"]]

        if not os.path.isfile(results_dir / "Xenium_FFPE_Human_Breast_Cancer_Preprocessed_Prealigned.tif"):
            img_he_upscaled = resize(img_he, 0.3637/0.2125, interpolation="cv2.INTER_CUBIC")
            img_he_pre_aligned = template_match(img_dapi, img_he_upscaled, matching="cv2.TM_CCOEFF", padding=100)
            tifffile.imwrite(results_dir / "Xenium_FFPE_Human_Breast_Cancer_Preprocessed_Prealigned.tif", img_he_pre_aligned)

        #for transform in transforms:
        #    test_wsireg(results_dir / "Xenium_FFPE_Human_Breast_Cancer_Preprocessed_Prealigned.tif",
        #                results_dir / "Xenium_DAPI_Level_0.tif", transform_list=transform, results_dir_=results_dir)

        # Test Alignment output
        for transform in transforms:
            wsireg_type = "-".join(transform)
            wsireg_dir = "wsireg_" + "_".join(transform)
            wsireg_registered_image = results_dir / wsireg_dir / "HE_registered.tif"
            img_he_registered = tifffile.imread(str(wsireg_registered_image))
            img_he_prealigned = tifffile.imread(str(results_dir / "Xenium_FFPE_Human_Breast_Cancer_Preprocessed_Prealigned.tif"))
            test_wsireg_alignments(img_he_registered, img_dapi, results_dir, wsireg_type)
