# Std
import os
from pathlib import Path

# Third Party
import cv2
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

from src.utils import (load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, load_image,
                       get_human_breast_he_resolution, get_human_breast_dapi_resolution)


# ---------------------------------------------------------- #
# HELPER METHODS


def gaussian_filter(img, sigma_, kernel_size_):
    """ Apply gaussian filtering to an image. kernel size and sigma_

    :param img: image input
    :param sigma_: standard deviation of gaussian distribution
    :param kernel_size_: the size of the kernel on which to apply
    :return: image filtered with gaussian blur
    """
    return cv2.GaussianBlur(img, (kernel_size_, kernel_size_), sigma_)


def resize(img_, scale_factor_, interpolation: str = "cv2.INTER_CUBIC"):
    """ Resize an image based on scale factor and interpolation method

    :param img_: the input image as numpy array.
    :param scale_factor_: the scale factor to which the image should be resized.
    :param interpolation: the interpolation method to resize image.
    :return: resized image.
    """

    interpolation = eval(interpolation)
    new_dim = (int(img_.shape[1]*scale_factor_), int(img_.shape[0]*scale_factor_))
    return cv2.resize(img_, new_dim, interpolation=interpolation)


def anisotropic_filter(img, weight_):
    """ Apply anisotropic filter called

    :param img:
    :param weight_:
    :return:
    """
    return denoise_tv_chambolle(img, weight_)


def template_match(img_dapi_, img_he_, matching: str, padding: int = 1000, convert_gray: bool = True):
    """ This method will use template matching to find the best match for an image. If specified, it will add more than
        the template matching such that it can be used for registration.

    :param img_dapi_: the image that you want to align on (fixed image)
    :param img_he_: the image that you want to align (moving image)
    :param matching: the method to use for matching
    :param padding: add padding to the bounding box
    :param convert_gray: convert the HE images to gray scale
    :return: the matched template of HE images onto DAPI image

    """
    matching = eval(matching)

    # Convert H&E to shades of gray
    if convert_gray:
        img_he_gray = cv2.cvtColor(img_he_, cv2.COLOR_BGR2GRAY)
    else:
        img_he_gray = img_he_

    # Scale dapi and invert
    img_dapi_scaled = cv2.normalize(img_dapi_, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img_dapi_invert = 255 - img_dapi_scaled

    # Template matching
    result = cv2.matchTemplate(img_he_gray, img_dapi_invert, matching)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    height, width = img_dapi_invert.shape[:2]

    # Return matches template
    img_he_aligned = crop_image(img_he_, top_left[1] - padding, top_left[0] - padding,
                                width + 2 * padding, height + 2 * padding)

    return img_he_aligned


def crop_image(img, x_, y_, width, height):
    """ This method crop the image based on a bounding box defined by (x, y, width, height)

    :param img:
    :param x_: vertical coordinate
    :param y_: horizontal coordinate
    :param width: horizontal length
    :param height: vertical length
    :return: cropped image
    """
    return img[x_:x_ + height, y_:y_ + width]


def flip_image(img, flip_code: int = 0):
    """ flip the image based on flip code

    :param img: image input
    :param flip_code: vertical or horizontal flipping (0: horizontal, 1: vertical)
    :return:
    """
    return cv2.flip(img, flip_code)


# -------------------------------------------------------------------- #
# Test Methods


def test_resize(img_, dapi_res_0: float = 0.2125, dapi_res_1: float = 0.425, he_res: float = 0.3637,
                results_dir_: Path = Path()):
    """
    This function test the result of the resize process. It tests up-sampling and down-sampling and with various
    methods and save the results in a folder called test_resize.

    :param img_: the input image
    :param dapi_res_0: the lower resolution to match
    :param dapi_res_1: the upper resolution to match
    :param he_res: the original resolution of the image
    :param results_dir_: the output directory where to save results
    :return: None
    """

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

            scale_factor_ = he_res / dapi_res

            sqr = 15
            axs[j, 0].imshow(img_[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
            axs[j, 0].set_title("Original")

            sqr = int(sqr * scale_factor)
            loc_ = [int(loc_[0] * scale_factor), int(loc_[1] * scale_factor_)]

            for i, method_ in enumerate(sampling_method):

                img_he_resized = resize(img_, scale_factor_, method_)

                # Visualize Up-sampling
                axs[j, i+1].imshow(img_he_resized[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
                axs[j, i+1].set_title(method_)

        plt.tight_layout()
        plt.savefig(test_resize_dir / f"{name}_sampling_comparison.png")
        plt.close(fig)


def test_noise_reduction(img_he_, results_dir_):
    """
    The goal is to visualize the impact of gaussian smoothing on the image if performed before or after resizing

    Nuclei are of size 6-10 microns
        - we have 0.36 microns/pixel <-> one nucleus is between 15-30 pixels
        - after upscaling: 0.2125 microns/pixel <-> one nucleus 30-50 pixels

    For gaussian smoothing, sigma is exerted on int(sigma*4) pixels:

    Resolution: 0.36
        - we want to impact 1/5 of the nuclei (between 3, 6 pixels)
        - Hence, we will test original resolution on sigma = [0.5, 1, 2, 3], kernel_size = 4
    Resolution: 0.2125
        - we want to impact 1/5 of the nuclei (between 6, 10 pixels)
        - sigma: 1 -> 4 pixels, sigma 3 -> 12 pixels
        - Hence, we will test on sigma = [1, 2, 3, 4] kernel_size = 8

    """

    gaussian_noise_dir = results_dir_ / "test_gaussian_noise"
    os.makedirs(gaussian_noise_dir, exist_ok=True)

    # Test Gaussian Blur Before Up-Sampling

    sigmas = [0.5, 1, 2, 3]
    kernel = 5
    method = "cv2.INTER_CUBIC"
    sqr = 20
    scale_factor_ = 0.3637 / 0.2125
    sqr_resized = int(sqr * scale_factor)

    locations = [[9443, 23183], [9364, 13652], [12028, 9948]]
    locations_resized = [[int(loc_[0] * scale_factor_), int(loc_[1] * scale_factor_)] for loc_ in locations]

    fig, axs = plt.subplots(nrows=len(locations), ncols=len(sigmas)+1, figsize=(30, 15))
    fig_resized, axs_resized = plt.subplots(nrows=len(locations), ncols=len(sigmas)+1, figsize=(30, 15))

    [ax.axis("off") for ax in axs.ravel()]
    [ax.axis("off") for ax in axs_resized.ravel()]

    axs[0, 0].set_title("Original")
    axs_resized[0, 0].set_title("Original")

    for i, sigma in enumerate(sigmas):
        img_he_filtered = gaussian_filter(img_he_, sigma, kernel)
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
    kernel = 7

    fig, axs = plt.subplots(nrows=len(locations), ncols=len(sigmas)+1, figsize=(30, 15))
    [ax.axis("off") for ax in axs.ravel()]

    for j, loc_ in enumerate(locations_resized):
        axs[j, 0].imshow(img_he_resized[loc_[0] - sqr_resized:loc_[0] + sqr_resized,
                                        loc_[1] - sqr_resized:loc_[1] + sqr_resized])

    axs[0, 0].set_title("Original")
    for i, sigma in enumerate(sigmas):
        img_he_filtered = gaussian_filter(img_he_resized, sigma, kernel)
        axs[0, i+1].set_title(f"sigma {sigma}")
        for j, loc_ in enumerate(locations_resized):
            axs[j, i+1].imshow(img_he_filtered[loc_[0]-sqr_resized:loc_[0]+sqr_resized,
                                               loc_[1]-sqr_resized:loc_[1]+sqr_resized])

    plt.tight_layout()
    fig.savefig(gaussian_noise_dir / "upsampled_image_gaussian_smoothing.png")
    plt.close(fig)


def test_anisotropic_filter(img_he_, results_dir_):
    """
    This function test the anisotropic filter before or after up-sampling and save the results in a folder called
    test_anisotropic_filter

    :param img_he_:
    :param results_dir_:
    :return: None
    """

    # img_he_ = cv2.cvtColor(img_he_, cv2.COLOR_BGR2GRAY)

    gaussian_noise_dir = results_dir_ / "test_anisotropic_filter"
    os.makedirs(gaussian_noise_dir, exist_ok=True)

    # Test Gaussian Blur Before Up-Sampling

    weights = [0.1, 0.3, 0.5]
    method = "cv2.INTER_CUBIC"
    sqr = 20
    scale_factor_ = 0.3637 / 0.2125
    sqr_resized = int(sqr * scale_factor)

    locations = [[1184, 217], [404, 3325], [1872, 2450]]
    locations_resized = [[int(loc_[0] * scale_factor_), int(loc_[1] * scale_factor_)] for loc_ in locations]

    fig, axs = plt.subplots(nrows=len(locations), ncols=len(weights)+1, figsize=(30, 15))
    fig_resized, axs_resized = plt.subplots(nrows=len(locations), ncols=len(weights)+1, figsize=(30, 15))

    [ax.axis("off") for ax in axs.ravel()]
    [ax.axis("off") for ax in axs_resized.ravel()]

    axs[0, 0].set_title("Original")
    axs_resized[0, 0].set_title("Original")

    for i, weight in enumerate(weights):
        img_he_filtered = denoise_tv_chambolle(img_he_, weight)
        img_he_filtered_norm = cv2.normalize(img_he_filtered, None, 0, 255, cv2.NORM_MINMAX)
        img_he_filtered_resized = (resize(img_he_filtered_norm, scale_factor_=scale_factor, interpolation=method)
                                   .astype(int))
        axs[0, i+1].set_title(f"iteration {weight}")
        axs_resized[0, i+1].set_title(f"iteration {weight}")
        for j, loc_ in enumerate(locations):
            axs[j, i+1].imshow(img_he_filtered_norm[loc_[0]-sqr:loc_[0]+sqr, loc_[1]-sqr:loc_[1]+sqr])
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


# -------------------------------------------------------------------- #
# Main preprocessing method


def preprocess_he(img_he_, img_dapi_, scale_factor_, resize_method_, kernel_, sigma_, match_method_, padding_):
    """ This function performs the preprocessing of H&E images. The parameters were tested on the Xenium FFPE Human
        Breast H&E additional image. This method performs 4 transformation:

        1. Flip the image vertically (alignment with corresponding DAPI)
        2. Resize the image with a scale factor (match resolution with corresponding DAPI)
        3. Gaussian noise filter to remove the noise linked with the camera
        4. Pre-template matching to DAPI with an additional padding of 1000 pixels

    :param img_he_: input raw h&e image
    :param img_dapi_: input raw dapi image
    :param scale_factor_: the factor to resize the image
    :param resize_method_: the method to resize the image
    :param sigma_: sigma used for gaussian blurring
    :param kernel_: kernel size for gaussian blurring
    :param match_method_: type of method
    :param padding_: number of pixels padded around matched area
    :returns: preprocessed image

    :return:
    """

    # 0. Initial crop to reduce computation
    x_, y_, height, width = 5680, 3795, 16977, 25150
    img_he_ = crop_image(img_he_, x_, y_, width=width, height=height)
    # 1. Flip the image
    img_he_ = flip_image(img_he_, flip_code=0)
    # 2. Resize according to scale factor and method
    img_he_ = resize(img_he_, scale_factor_, resize_method_)
    # 3. Denoising with Gaussian filtering
    img_he_ = gaussian_filter(img_he_, kernel_, sigma_)
    # 4. Match filtered image with corresponding DAPI image
    img = template_match(img_dapi_, img_he_, match_method_, padding_)

    return img


def build_results_dir():
    results_path = get_results_path()
    results_path = results_path / "he_preprocessing"
    os.makedirs(results_path, exist_ok=True)
    return results_path


if __name__ == "__main__":

    # ------------------------------- #
    # Script Parameters

    run_preprocessing = False
    run_tests = True
    he_level = 0
    dapi_level = 0
    img_type = "mip"

    # ------------------------------- #
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

    if run_preprocessing:
        print("Preprocessing DAPI Image")

    elif run_tests:
        print("Test Preprocessing Method")
        # Test Up-sampling vs Down-sampling
        test_resize(img_he, results_dir_=results_dir)

        # Test Different Noise Reduction + before vs after up-sampling
        test_noise_reduction(img_he, results_dir_=results_dir)
        sub_image = img_he[img_he.shape[0] // 2 - 1000:img_he.shape[0] // 2 + 1000,
                           img_he.shape[1] // 2-2000:img_he.shape[1] // 2 + 2000,]
        test_anisotropic_filter(sub_image, results_dir_=results_dir)

