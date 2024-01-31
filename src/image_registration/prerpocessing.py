"""
In this image we will preprocess both images to align them with one another



"""

import cv2
import os
from pathlib import Path

from src.utils import load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, load_image

# -------------------------------------- #
# HELPER METHODS


def resize(img_he, scale_factor_, interpolation: str = "cv2.INTERCUBIC"):
    """ Resize an image based on scale factor and interpolation method """

    interpolation = eval(interpolation)
    new_dim = (int(img_he.shape[1]*scale_factor_), int(img_he.shape[0]*scale_factor_))
    return cv2.resize(img_he, new_dim, interpolation=interpolation)


def template_match(img_dapi, img_he, matching):
    matching = eval(matching)

    # Convert H&E to shades of gray
    img_he_gray = cv2.cvtColor(img_he, cv2.COLOR_BGR2GRAY)

    # Scale dapi and invert
    img_dapi_scaled = cv2.normalize(img_dapi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img_dapi_invert = 255 - img_dapi_scaled

    # Template matching
    result = cv2.matchTemplate(img_he_gray, img_dapi_invert, matching)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    height, width = img_dapi_invert.shape[:2]

    # Return matches template
    img_he_aligned = crop_image(img_he, top_left, width, height)

    return img_he_aligned


def crop_image(img, top_left, width, height):
    return img[top_left[1]:top_left[1] + height, top_left[0]:top_left[0] + width]


def flip_image(img):
    return cv2.flip(img, 0)


# -------------------------------------- #
# TEST METHODS


def test_resize(dapi_res_0: float = 0.2125, he_res: float = 0.3637, results_dir_: Path = Path()):
    """ TODO: solve the issue with alignment -> needs to compare after alignment ? """

    test_resize_dir = results_dir_ / "test_resize"
    os.makedirs(test_resize_dir)

    he_img = load_xenium_he_ome_tiff(hb_path, level_=0)

    print(f"Testing: up-sampling methods on H&E level 0 ({he_res} -> {dapi_res_0}")

    dapi_img_0 = load_image(get_human_breast_he_path(), img_type="mip", level_=0)

    he_res = 0.3637
    dapi_res_0 = 0.2125

    scale_factor = he_res / dapi_res_0

    up_sampling_method = ["cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_LANCZOS4"]

    for method_ in up_sampling_method:
        img_he_resized = resize(he_img, scale_factor, method_)

    dapi_img_1 = load_image(get_human_breast_he_path(), img_type="mip", level_=1)


# -------------------------------------- #
# MAIN METHODS

def preprocess_he_image():
    """
    Descr: The original h&e image contains the xenium panels and is vertically flipped compared to DAPI this method
           flip and crop the image based on value visually determined.

    -  1. Crop image with a bounding box (x, y, w, h)
    -  2. flip image and save it under a new format
    """

    pass


def alignment():
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
    he_level = 0  # resolution: 0.3638 microns / pixel
    run_tests = True
    run_alignment = False

    # ---------------------------------------- #
    # General Predefined parameters
    hb_path = get_human_breast_he_path()
    results_dir = build_results_dir()

    # ---------------------------------------- #
    # Main Methods

    preprocess_he_image()

    if run_alignment:
        pass
        # Step 1: Ensure that both images have the same resolution

        # Step 2: Apply noise removal

        # Step 3: Invert DAPI image to have similar

    # ---------------------------------------- #

    if run_tests:
        # Test UP-SAMPLING versus DOWN-SAMPLING at different location
        # Methods:
        #   - compare with DAPI same location but different methods of up-sampling or down-sampling

        pass
