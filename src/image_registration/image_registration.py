"""
In this image we will preprocess both images and align them with one another


Strategy for image registration:
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

Strategy:
- Run stardist on H&E and DAPI to observe performance
- Register DAPI / H&E to have a measurable comparison
- Compute ground truth on 3 tiles (n x n) and then measure the accuracy of both models
    - the ground truth should represent DAPI and H&E signal -> create a good protocol for this

Implementations:
[ x ]: Test different methods for up-scaling and down-scaling and choose one
       -> visually up-scaling seems to not introduce significant artifact
       -> selected method for interpolation = cv2.INTER_CUBIC
[ x ]: Test gaussian smoothing with various sigma before or after upscaling
       -> compute sigma based on estimated nuclei size since this is what we would like to segment
       ->
[ x ]: Run test with WSIREG library:
       - run wsireg and then run stardist to check the alignment
       - in addition compare this to the stardist on DAPI to observe how it performs

[ x ]: Create the main pipeline for image registration
[ x ]: Clean the code

"""

# Std library
import pickle
from typing import Optional
import os
from pathlib import Path

# Third parties
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac
#from wsireg.wsireg2d import WsiReg2D

# Relative
from src.utils import (load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, load_image,
                       get_human_breast_he_resolution, get_human_breast_dapi_resolution)
from src.dapi_preprocessing.dapi_preprocessing import preprocess_dapi
from src.nucleus_segmentation.segmentation_stardist import build_stardist_mask_outlines
from src.he_preprocessing.he_preprocessing import preprocess_he
# -------------------------------------- #
# HELPER METHODS


def segment_stardist(
        img: np.ndarray,
        model_type_: str,
        prob_thrsh: Optional[float],
        nms_thrsh: Optional[float],
        n_tiles: tuple,
):
    model = StarDist2D.from_pretrained(model_type_)

    img_normalized = normalize(img, 1, 99, axis=(0, 1))
    labels, details = model.predict_instances(img_normalized, prob_thresh=prob_thrsh, nms_thresh=nms_thrsh,
                                              n_tiles=n_tiles)

    return labels, details

# -------------------------------------- #
# TEST METHODS





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

def preprocess_dapi_image(img_dapi_: np.ndarray, save_img: Path):
    """

    :param img_dapi_:
    :param save_img:
    :return:
    """
    save_img_path = save_img / "Xenium_FFPE_Human_Breast_Cancer_DAPI-MIP_Preprocessed.tif"
    if os.path.isfile(save_img_path):
        return tifffile.imread(str(save_img_path)), save_img_path

    # Based on Preprocessing Test in dapi_preprocessing.py
    img_dapi_ = preprocess_dapi(img_dapi_, method_="rolling_ball", radius_=30, sigma_=4)

    if save_img is not None:
        tifffile.imwrite(save_img_path, img_dapi_)
        return img_dapi_, save_img_path

    return img_dapi_


def preprocess_he_image(img_he_, img_dapi_, save_img: Optional[Path] = None):
    """
    Descr: The original h&e image is preprocessed such that it can be used for registration as follows:

    :param img_dapi_: input raw dapi image
    :param img_he_: input raw h&e image
    :param save_img: (Optional) path to save the final image
    :returns: preprocessed image
    """

    save_img_path = save_img / "Xenium_FFPE_Human_Breast_Cancer_HE_Preprocessed.tif"
    if os.path.isfile(save_img_path):
        return tifffile.imread(str(save_img_path)), save_img_path

    # Based on Preprocessing Test in he_preprocessing.py
    match_method_ = "cv2.TM_CCOEFF"
    resize_method_ = "cv2.INTER_CUBIC"
    sigma_ = 1
    kernel_ = 7
    padding_ = 1000
    scale_factor_ = 0.3637 / 0.2125
    img_he_preprocessed = preprocess_he(img_he_, img_dapi_, scale_factor_, resize_method_, kernel_, sigma_,
                                        match_method_, padding_)

    if save_img is not None:
        tifffile.imwrite(save_img_path, img_he_preprocessed)
        return img_dapi_, save_img_path

    return img_dapi_


def quality_check(img_he_: np.ndarray, img_dapi_: np.ndarray, locations_: list, square_size_: int, results_dir_):

    # Create directories
    wsireg_alignment_path = results_dir_ / "qc_registration"
    os.makedirs(wsireg_alignment_path, exist_ok=True)

    # Store Masks for each registration
    mask_dir = wsireg_alignment_path / "masks"
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = mask_dir / f"masks_registrations.pkl"

    # Create masks for He registered
    if not os.path.isfile(mask_path):
        model_type_he = "2D_versatile_he"
        labels_he, masks_he = segment_stardist(img_he_, model_type_he, prob_thrsh=None, nms_thrsh=None,
                                               n_tiles=(5, 5, 1))
        with open(mask_path, 'wb') as file:
            pickle.dump(masks_he, file)

    else:
        with open(mask_path, 'rb') as file:
            masks_he = pickle.load(file)

    masks_he = build_stardist_mask_outlines(masks_he["coord"])

    # Create masks for Preprocessed DAPI
    dapi_mask_path = results_dir_ / "masks" / "results_stardist_dapi_0_nms-None_thrsh-None.pkl"
    if not os.path.isfile(dapi_mask_path):
        model_type_dapi = "2D_versatile_fluo"
        labels_dapi, masks_dapi = segment_stardist(img_dapi_, model_type_dapi, prob_thrsh=None, nms_thrsh=None,
                                                   n_tiles=(5, 5))
        with open(dapi_mask_path, 'wb') as file:
            pickle.dump(masks_dapi, file)
    else:
        with open(dapi_mask_path, "rb") as file:
            masks_dapi = pickle.load(file)

    masks_dapi = build_stardist_mask_outlines(masks_dapi["coord"])

    fig, axs = plt.subplots(nrows=3, ncols=len(locations_), figsize=(20, 10))
    [ax.axis("off") for ax in axs.ravel()]

    for i, loc_ in enumerate(locations_):
        x_range = [loc_[0]-square_size_, loc_[0]+square_size_]
        y_range = [loc_[1]-square_size_, loc_[1]+square_size_]
        axs[0, i].imshow(img_he_[x_range[0]: x_range[1], y_range[0]: y_range[1]])
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
    plt.savefig(wsireg_alignment_path / f"alignment.png")


def compare_he_dapi_segmentation(img_dapi_, img_he_, ground_truth_, results_dir_):
    """ This function will compare stardist prediction on registered/preprocessed H&E and DAPI.
        For comparison purpose this will be measured compared to a ground truth on 3 different tiles.
        These tiles were manually annotated by using DAPI and HE-hematoxylin-deconvolved

    :param img_dapi_:
    :param img_he_:
    :param ground_truth_:
    :param results_dir_:
    :return:
    """

    # Bounding Boxes in Pixel

def registration_intensity_based(img_he_path_, img_dapi_path_, transformations_list_, results_dir_):
    """ ITK registration is intensity based and performs well for multimodal registration

    :param img_he_path_: moving image
    :param img_dapi_path_: fixed image
    :param transformations_list_:
    :param results_dir_:
    :return:
    """
    save_img_name = "Xenium_FFPE_Human_Breast_Cancer_DAPI_registered.tif"

    if os.path.isfile(results_dir_ / save_img_name):
        return tifffile.imread(results_dir_ / save_img_name)

    transform_ = "wsireg_" + "_".join(transformations_list_)
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
        reg_params=transformations_list_
    )

    reg_graph.register_images()

    reg_graph.save_transformations()

    reg_graph.transform_images(file_writer="ome.tiff")

    # Aligned files
    output_file = wsireg_results / "registration-DAPI_to_HE_registered.ome.tiff"
    img_he_registered_ = tifffile.imread(str(output_file))
    tifffile.imwrite(results_dir_ / save_img_name, img_he_registered_)

    return img_he_registered_


def registration_feature_based(img_he_, img_dapi_):
    """
    Description: the goal here is to test feature alignment. However, we need to identify these features
        1. Run Stardist on both images
        2. Do a rough alignments and find nuclei that are close to one another


    """
    raise ValueError("Not Implemented")


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
    # Parameters: Predefined
    hb_path = get_human_breast_he_path()
    results_dir = build_results_dir()
    scale_factor = get_human_breast_he_resolution() / get_human_breast_dapi_resolution()

    # ---------------------------------------- #
    # Main Methods

    img_he_path = get_human_breast_he_path() / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"
    img_he, metadata = load_xenium_he_ome_tiff(img_he_path, level_=he_level)
    img_dapi_path = get_human_breast_he_path() / "morphology_mip.ome.tif"
    img_dapi = load_image(get_human_breast_he_path(), level_=dapi_level, img_type=img_type)

    # Save original DAPI images
    img_dapi_0_path = results_dir / "Xenium_DAPI_Level_0.tif"
    tifffile.imwrite(img_dapi_0_path, img_dapi)

    # Preprocess and save H&E image
    img_he_pre, img_he_pre_path = preprocess_he_image(img_he, img_dapi, save_img=results_dir)
    img_dapi_pre, img_dapi_pre_path = preprocess_dapi_image(img_dapi, save_img=results_dir)

    if run_alignment:
        transforms = ["rigid", "affine"]
        img_registered = registration_intensity_based(img_dapi_pre_path, img_he_pre_path, transforms, results_dir)

        # Different positions to show the results of alignments
        locations = [[img_registered.shape[0] // 2, img_registered.shape[1] // 2], [9550, 7320], [10470, 17540],
                     [1000, 1000], [img_registered.shape[0] - 500, img_registered[1].shape[1] - 500]]

        quality_check(img_registered, img_dapi_pre, locations_=locations, square_size_=200, results_dir_=results_dir)
    # ---------------------------------------- #

    if run_tests:

        # Test registration wsireg. This requires the preprocessed image
        transforms = [["rigid"], ["rigid", "affine"], ["rigid", "affine", "nl"], ["affine", "nl"]]
        for transform in transforms:
            test_wsireg(img_he_pre_path, img_dapi_0_path, transform_list=transform, results_dir_=results_dir)

        # Test registrations output.
        for transform in transforms:
            wsireg_type = "-".join(transform)
            wsireg_dir = "wsireg_" + "_".join(transform)
            wsireg_registered_image = results_dir / wsireg_dir / "HE_registered.tif"
            img_he_registered = tifffile.imread(str(wsireg_registered_image))
            test_wsireg_alignments(img_he_registered, img_dapi, results_dir, wsireg_type)
