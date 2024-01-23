# Std Library
import os
from pathlib import Path

# Third Party
import numpy as np
import cv2
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.restoration import rolling_ball

# Relative Imports
from src.utils import get_mouse_xenium_path, load_image, get_results_path, image_patch

RESULTS = Path()


def noise_reduction(img_, sigma_):
    return gaussian_filter(img_, sigma=sigma_)


def background_gaussian_estimation(img_, sigma_=50):
    """
    Estimate the background of an image using Gaussian blurring.
    :param img_: Input image.
    :param sigma_: Standard deviation.
    :return: Estimated background.
    """
    background = gaussian_filter(img_.astype(float), sigma=sigma_)
    return background


def background_rolling_ball_estimation(img_, radius_):
    """
    Estimate the background of an image using Rolling Ball algorithm.
    :param img_: Input image.
    :param radius_: radius.
    :return: Estimated background.
    """
    return rolling_ball(img_, radius=radius_)


def background_top_hat(img_, kernel_size=5, iterations=1):
    """
    Applies a top-hat filter to an input image.

    Parameters:
        - input_image: The input image to which the filter will be applied.
        - kernel_size: The size of the kernel used for the filtering (default is 5).
        - iterations: The number of times the operation will be applied (default is 1).

    Returns:
        - The filtered image.
    """
    # Create a rectangular kernel for the top-hat operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply the top-hat filter
    top_hat_result = cv2.morphologyEx(img_, cv2.MORPH_TOPHAT, kernel, iterations=iterations)

    return top_hat_result


def remove_background(img_, method_: str = "gaussian", radius_: int = 5, top_hat_kernel_: int = 5,
                      top_hat_iter_: int = 1, sigma_bg_: int = 3):
    """
    Remove the estimated background from the image.
    :param img_: Input image.
    :param method_: method used for background estimation (gaussian, rolling_ball, top_hat)
    :param radius_: Size of the Gaussian filter for background estimation.
    :param top_hat_iter_: parameter used for top hat algorithm
    :param top_hat_kernel_: parameter used for top hat algorithm
    :param sigma_bg_: parameter used for gaussian background estimation
    :return: Image with background removed.
    """
    if method_ == "gaussian":
        background = background_gaussian_estimation(img_, sigma_=sigma_bg_)
    elif method_ == "rolling_ball":
        background = background_rolling_ball_estimation(img_, radius_=radius_)
    elif method_ == "top_hat":
        background = background_top_hat(img_, kernel_size=top_hat_kernel_, iterations=top_hat_iter_)
    else:
        raise ValueError("Invalid method for background removal")

    img_no_bg = img_ - background

    return img_no_bg.astype(img_.dtype)


def preprocess_dapi(img_: np.ndarray, method_, sigma_, radius_: int = 30, top_hat_kernel: int = 5, top_hat_iter: int = 1):

    # Step 1: Noise reduction
    img_ = noise_reduction(img_, sigma_=sigma_)

    # Step 2: Remove Background
    img_ = remove_background(img_, method_=method_, radius_=radius_,
                             top_hat_iter_=top_hat_iter, top_hat_kernel_=top_hat_kernel)

    return img_


def build_result_dir():
    global RESULTS
    RESULTS = get_results_path() / "dapi_preprocessing"

    os.makedirs(RESULTS, exist_ok=True)


if __name__ == "__main__":

    build_result_dir()

    mouse_replicate_path = get_mouse_xenium_path()
    img = load_image(mouse_replicate_path, img_type="mip", level_=0)
    img, _ = image_patch(img, square_size_=200)

    img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_image = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(str(RESULTS / f"original.jpg"), colored_image)

    # Characterization of an acceptable sigma
    # 1 pixel = 0.2125 microns and 1 microns = 1 / 0.2125 = 4.7 pixels
    # For the mouse brain the nucleus is around 5-10 microns in diameter
    # This means between 23 and 47 microns for one nucleus
    # We start with 1/5 of the nucleus 4 to 10 kernel size

    sigmas = [2, 3, 4, 5]
    # radius = round(4.0 x sigma) hence a good estimation is between 2-5 for our gaussian filtering
    for sig in sigmas:
        img_new = noise_reduction(img, sigma_=sig)
        img_normalized = cv2.normalize(img_new, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored_image = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(RESULTS / f"noise_reduction_{sig}.jpg"), colored_image)

    # Based on optimization
    sigma = 4
    method = "rolling_ball"
    radiuses = [10, 20, 30, 50, 70, 100]
    for rad in radiuses:
        img_rad = preprocess_dapi(img, sigma_=sigma, method_=method, radius_=rad)
        img_normalized = cv2.normalize(img_rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored_image = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(RESULTS / f"dapi_preprocessed_{method}_sigma{sigma}_radius{rad}.jpg"), colored_image)

    kernel = [1, 2, 4, 5, 7, 10]
    iteration = [1, 2, 3, 4, 5]
    method = "top_hat"
    for ker in kernel:
        for iter_ in iteration:
            img_top_hat = preprocess_dapi(img, sigma_=sigma, method_="top_hat", top_hat_kernel=ker, top_hat_iter=iter_)
            img_normalized = cv2.normalize(img_top_hat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored_image = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
            cv2.imwrite(str(RESULTS / f"dapi_preprocessed_{method}_sigma{sigma}_kernel{ker}_iter{iter_}.jpg"), colored_image)

    # Seems to be a good preprocessing pipeline
    radius = 30
    method = "rolling_ball"
    img = load_image(mouse_replicate_path, img_type="mip", level_=0)
    img = preprocess_dapi(img, sigma_=sigma, method_=method, radius_=radius)
    tifffile.imwrite(RESULTS / "mouse_replicate_dapi_preprocessed.tif", img)




