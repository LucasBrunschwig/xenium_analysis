
# Std Library
import os
import pickle
import platform
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Relative Imports
from src.utils import load_xenium_he_ome_tiff, get_human_breast_he_path, get_results_path, load_image

RESULTS = Path()


def relative_difference(list1, list2):
    matrix = []
    for x in list1:
        row = []
        for y in list2:
            diff = abs(x - y) / max(abs(x), abs(y))
            row.append(diff)
        matrix.append(row)
    return np.array(matrix)


def template_matching(img_dapi, img_he, method_, scale_factor_):
    method_f = eval(method_)

    # Convert HE to gray scale
    img_he_gray = cv2.cvtColor(img_he, cv2.COLOR_BGR2GRAY)
    img_he_gray = cv2.flip(img_he_gray, 0)
    new_dim = (int(img_he_gray.shape[1]*scale_factor_), int(img_he_gray.shape[0]*scale_factor_))
    img_he_gray = cv2.resize(img_he_gray, new_dim, interpolation=cv2.INTER_CUBIC)
    img_he = cv2.resize(img_he, new_dim, interpolation=cv2.INTER_CUBIC)

    # Normalizing and Inverting DAPI
    img_dapi_scaled = cv2.normalize(img_dapi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img_dapi_invert = 255 - img_dapi_scaled

    # Template Matching
    result = cv2.matchTemplate(img_he_gray, img_dapi_invert, method_f)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Top left corner of the matched area
    top_left_ = max_loc
    height, width = img_dapi_invert.shape[:2]

    img_he_aligned_ = cv2.flip(img_he, 0)[top_left_[1]:top_left_[1]+height, top_left_[0]:top_left_[0]+width]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    [ax.axis("off") for ax in axs]
    axs[0].imshow(img_dapi)
    axs[1].imshow(img_he_aligned_)
    plt.tight_layout()
    plt.savefig(RESULTS / f"alignment_{method_}.png")

    if "COEFF" in method_:
        cv2.imwrite(str(RESULTS / f"Human_Breast_Cancer_{method_}_he_image_crop_scale.png"), img_he_aligned_)

    return img_he_aligned_, top_left_


def compare_nuclei(img_he_, img_he_cropped_scaled_, img_he_aligned_, scale_factor_, crop_position, transform_,
                   positions_, square_size_):
    """
    The goal of this function is to compare both image at similar location to observe how the nuclei shape
    is impacted by the various transformation. This is important since we are also interested in nuclear
    morphology.
    """

    fig, axs = plt.subplots(nrows=len(positions_), ncols=3, figsize=(30, 5*len(positions_)))

    img_he_ = cv2.flip(img_he_, flipCode=0)

    for i, position in enumerate(positions_):

        indices = 0 if len(positions_) == 1 else (i, 0)
        axs[indices].imshow(img_he_[position[0]-square_size_:position[0]+square_size_,
                                    position[1]-square_size_:position[1]+square_size_])

        axs[indices].set_title("Original Image")

        # Convert Position to the scaled and cropped image
        position_scaled = [int(pos*scale_factor_) for pos in position]
        position_cropped = position_scaled
        position_cropped[0] = position_cropped[0] - crop_position[1]
        position_cropped[1] = position_cropped[1] - crop_position[0]
        square_size_scaled = int(square_size_*scale_factor_)

        indices = 1 if len(positions_) == 1 else (i, 1)
        axs[indices].imshow(img_he_cropped_scaled_[position_cropped[0] - square_size_scaled:position_cropped[0] + square_size_scaled,
                                                   position_cropped[1] - square_size_scaled:position_cropped[1] + square_size_scaled])
        axs[indices].set_title("Cropped-Scaled Image")

        transform_matrix = np.array([[0.999466604064691, 0.005119841279342, -31.485396983163184],
                                     [-0.005024121311802, 1.000125672401803, 22.629295815166188]])

        position_aligned = [int(el) for el in np.matmul(transform_matrix, np.array(position_cropped + [1]))]

        indices = 2 if len(positions_) == 1 else (i, 2)
        axs[indices].imshow(img_he_aligned_[position_aligned[0] - square_size_scaled:position_aligned[0] + square_size_scaled,
                                            position_aligned[1] - square_size_scaled:position_aligned[1] + square_size_scaled])
        axs[indices].set_title("Aligned Image")

    plt.tight_layout()
    plt.savefig(RESULTS / "comparison_before_after_registration.png")




def build_results_dir():
    global RESULTS
    RESULTS = get_results_path() / "he_alignment"
    os.makedirs(RESULTS, exist_ok=True)


if __name__ == "__main__":

    # Scripts Parameters
    # ----------------------------------

    level_he = 0
    level_dapi = 1

    # ----------------------------------

    build_results_dir()

    print("# ----------------------------------- #")

    print("Analyzing the resolution")
    dapi_res = [0.2125*i for i in range(1, 5)]
    he_res = [0.36378*i for i in range(1, 5)]
    print("\tDAPI", dapi_res)
    print("\tHE", he_res)
    matrix = relative_difference(dapi_res, he_res)
    min_indices = np.unravel_index(np.argmin(matrix), matrix.shape)
    print(f"\tMinimum relative difference (1): DAPI - {min_indices[0]}, HE - {min_indices[1]} / {np.min(matrix):.2f}")
    matrix[min_indices] = np.inf
    min_indices = np.unravel_index(np.argmin(matrix), matrix.shape)
    print(f"\tMinimum relative difference (2): DAPI - {min_indices[0]}, HE - {min_indices[1]} / {np.min(matrix):.2f}")

    print("# ----------------------------------- #")

    # Load ome-tiff images
    human_breast_path = get_human_breast_he_path()
    he_ome_tiff = human_breast_path / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"

    # Use custom loading for H&E image and metadata
    print("Loading H&E...")
    image_he, metadata = load_xenium_he_ome_tiff(he_ome_tiff, level_=level_he)
    print("metadata", metadata)
    print(f"Resolution {level_he}: {metadata['x_size']*(level_he+1)}")
    print()

    # Use custom loading for DAPI
    print("Loading DAPI...")
    image_dapi = load_image(human_breast_path, img_type="mip", level_=level_dapi)
    print("metadta: x_size: 0.2125, y_size: 0.2125")
    print(f"Resolution {level_dapi}: {0.2125*(level_dapi+1)}")

    print("# ----------------------------------- #")

    # Perform template matching with various strategy
    print("Template Matching and Cropping")
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
               'cv2.TM_SQDIFF_NORMED']
    scale_factor = 0.363788 / 0.425

    # for method in methods:
    #    print(f"Method {method}")
    #    img_he_cropped_scaled, top_left = template_matching(image_dapi, image_he, method, scale_factor)

    print("\tSelected_method: cv2.TM_CCOEFF")
    image_he_cropped_scaled, top_left = template_matching(image_dapi, image_he, 'cv2.TM_CCOEFF', scale_factor)

    print("# ----------------------------------- #")

    print("Comparing H&E before and after registration")

    # Here we would like to check for an image how the rescaling and then registration affects the nuclei shape

    positions = [[12000, 15000], [9000, 20000], [12000, 8000]]

    print(f"\tpositions {positions}")

    square_size = 300
    img_he_aligned = cv2.imread(str(RESULTS / "Human_Breast_Replicate1_HE_aligned.tif"))
    transform = None
    compare_nuclei(image_he, image_he_cropped_scaled, img_he_aligned, scale_factor, top_left, transform,
                   positions, square_size)

    print("# ----------------------------------- #")












