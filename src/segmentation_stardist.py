# Std
import os
import pickle
from pathlib import Path

# Third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import platform
from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize
from itertools import product


# Relative import
from utils import load_image, image_patch, load_xenium_data


if platform.system() != "Windows":
    import resource
    # Set the maximum memory usage in bytes (300GB)
    max_memory = int(3e11)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))

RESULTS = Path()
RESULTS_3D = Path()

if torch.cuda.is_available():
    print("GPU available", torch.cuda.current_device())
else:
    print("No GPU available")


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = True


def get_xenium_nucleus_boundaries(path_replicate_: Path, boundaries):

    adata = load_xenium_data(Path(str(path_replicate_) + ".h5ad"))

    # Convert xenium predefined nucleus boundaries to pixels locations
    # (x,y): vertex_x is the horizontal axis / vertex y is the vertical axis
    # from_metadata 1 pixel = 0.2125 microns
    x_conversion = 0.2125
    y_conversion = 0.2125
    adata.uns["nucleus_boundaries"]["vertex_y_pixel"] = adata.uns["nucleus_boundaries"]["vertex_y"].apply(
        lambda p: round(p/y_conversion))
    adata.uns["nucleus_boundaries"]["vertex_x_pixel"] = adata.uns["nucleus_boundaries"]["vertex_x"].apply(
        lambda p: round(p/x_conversion))

    # Selection of segmented nucleus that are inside the patch
    pix_boundaries = adata.uns["nucleus_boundaries"][(adata.uns["nucleus_boundaries"]["vertex_x_pixel"] > boundaries[1][0]) &
                                                     (adata.uns["nucleus_boundaries"]["vertex_x_pixel"] < boundaries[1][1]) &
                                                     (adata.uns["nucleus_boundaries"]["vertex_y_pixel"] > boundaries[0][0]) &
                                                     (adata.uns["nucleus_boundaries"]["vertex_y_pixel"] < boundaries[0][1])
                                                     ]
    masks = [[], []]
    for cell_seg in pix_boundaries["cell_id"].unique():
        masks[0].append(pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy() - boundaries[1][0])
        masks[1].append(pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy() - boundaries[0][0])

    return masks


def segment_stardist(
        img: np.ndarray,
        model_type_: str,
        do_3d: bool = False,
        prob_thrsh: float = None,
        nms_thrsh: float = None,
) -> tuple:
    """Run cellpose and get masks

    Parameters
    ----------
    img: Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type_: pretrained models 2D_versatile_fluo / 2D_paper_dsb2018, 2D_versatile_he
    do_3d: perform 3D nuclear segmentation requires 3D array

    Returns
    -------
    np.ndarray
        labelled image, where 0 = no masks; 1, 2, ... = mask labels
    """
    if not do_3d:
        model = StarDist2D.from_pretrained(model_type_)
        # normalizer (perform normalization), axes, sparse (aggregation), prob_thresh, nms_thresh (non-maximum suppression), scale (factor), n_tiles (broken up in overl
        labels, details = model.predict_instances(normalize(img), prob_thresh=prob_thrsh, nms_thresh=nms_thrsh)
        coord = details["coord"]
    else:
        model = StarDist3D.from_pretrained(model_type_)
        labels, details = model.predict_instances(normalize(img))
        coord = details["points"]

    return labels, coord


def optimize_stardist_2d(path_replicate_: Path, model_type_: str, image_type_: str, square_size: int):

    img = load_image(path_replicate_, img_type=image_type_)
    patch, boundaries = image_patch(img, square_size=square_size)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 40))
    [ax.axis("off") for ax in axs.ravel()]
    [ax.imshow(patch) for ax in axs.ravel()]

    prob_thresh = [0.3, 0.4, 0.5, 0.7]
    nms_thresh = [0.3, 0.5, 0.7, 0.9]
    comb = product(prob_thresh, nms_thresh)
    for ax, (prob, nms) in zip(axs.ravel(), comb):
        labels, masks_stardist = segment_stardist(patch, model_type_=model_type_, do_3d=False,
                                                  prob_thrsh=prob, nms_thrsh=nms)
        ax.set_title(f"Prob: {prob}, Nms: {nms}")
        for mask in masks_stardist:
            closed_contour = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
            ax.plot(closed_contour[1, :], closed_contour[0, :], 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(RESULTS / f"stardist_2d_optimization_{image_type_}_{model_type_}_{square_size}.png")
    plt.close()


def run_patch_stardist_2d(path_replicate_: Path, model_type_: str, image_type_: str):

    img = load_image(path_replicate_, img_type=image_type_)
    initial = 400
    square_sizes = [400, 600, 800, 1000, 1600, 2000, 4000, 8000]

    patch, _ = image_patch(img, square_size=square_sizes[0])

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 30))
    [ax.axis("off") for ax in axs.ravel()]
    [ax.imshow(patch) for ax in axs.ravel()]

    for ax, size in zip(axs.ravel(), square_sizes):
        patch, boundaries = image_patch(img, square_size=size)

        labels, masks_stardist = segment_stardist(patch, model_type_=model_type_, do_3d=False,
                                                  prob_thrsh=None, nms_thrsh=None)

        ax.set_title(f"Square Size = {size}")
        center_x = patch.shape[0] // 2
        center_y = patch.shape[1] // 2
        min_ = center_x - initial // 2
        max_ = center_x + initial // 2
        bottom_left = center_y - initial // 2
        bottom_right = center_y + initial // 2
        for mask in masks_stardist:
            if ((max_ > mask[1, :].max() > min_) or
                (max_ > mask[1, :].min() > min_)) \
               and ((max_ > mask[0, :].max() > min_) or
                    (max_ > mask[0, :].min() > min_)):
                closed_contour = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
                closed_contour[0, :] = closed_contour[0, :] - (patch.shape[0] // 2 - initial // 2)
                closed_contour[1, :] = closed_contour[1, :] - (patch.shape[1] // 2 - initial // 2)
                ax.plot(closed_contour[1, :], closed_contour[0, :], 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(RESULTS / f"stardist_2d_optimization_{image_type_}_{model_type_}_patches_unnormalized.png")
    plt.close()


def run_stardist_2d(path_replicate_: Path, model_type_: str, image_type_: str,
                    prob_thrsh: float = None, nms_thrsh: float = None, square_size: int = 400):

    img = load_image(path_replicate_, img_type=image_type_)
    patch, boundaries = image_patch(img, square_size=square_size)

    labels, masks_stardist = segment_stardist(patch, model_type_=model_type_, do_3d=False,
                                              prob_thrsh=prob_thrsh, nms_thrsh=nms_thrsh)
    masks_xenium = get_xenium_nucleus_boundaries(path_replicate_, boundaries)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    [ax.axis("off") for ax in axs]
    [ax.imshow(patch) for ax in axs]
    axs[0].set_title("Original Image")
    axs[2].set_title("Stardist Segmentation")
    for mask in masks_stardist:
        closed_contour = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
        axs[2].plot(closed_contour[1, :], closed_contour[0, :], 'r', linewidth=.8)

    axs[1].set_title("Xenium Segmentation")
    for x, y in zip(masks_xenium[0], masks_xenium[1]):
        axs[1].plot(x, y, 'r', linewidth=.8)

    plt.savefig(RESULTS / f"stardist_2d_{image_type_}_{model_type_}_{square_size}.png")
    plt.close()


def run_stardist_3d(path_replicate_: Path, model_type_: str, level_: int = 0, diameter_: int = 10):

    img = load_image(path_replicate_, img_type="stack", level_=level_)
    patch, boundaries = image_patch(img, square_size=700)

    fig, axs = plt.subplots(3, 4)

    for i, (layer, ax) in enumerate(zip(patch, axs.ravel())):
        ax.axis("off")
        ax.set_title(f"patch - layer {i}")
        ax.imshow(layer)

    plt.tight_layout()
    fig.savefig(RESULTS_3D / f"3d_patch_og_level{level_}_diameter{diameter_}.png", dpi=600)

    print("Segmenting the whole image")

    if not os.path.isfile(RESULTS_3D / f"stardist_mask_level{level_}.pkl"):
        seg_3d, coord = segment_stardist(patch, model_type_=model_type_, do_3d=True)
        with open(RESULTS_3D / f"stardist_mask_level{level_}.pkl", "wb") as file:
            pickle.dump(seg_3d, file)
    else:
        with open(RESULTS_3D / f"stardist_mask_level{level_}.pkl", "rb") as file:
            seg_3d = pickle.load(file)

    print("Plotting Resulting Segmentation")

    fig, axs = plt.subplots(3, 4)

    for i, (layer, ax) in enumerate(zip(patch, axs.ravel())):
        ax.axis("off")
        ax.set_title(f"stardist - layer {i}")
        ax.imshow(layer)
        ax.imshow(seg_3d[i, :, :])

    plt.tight_layout()
    fig.savefig(RESULTS_3D / f"3d_patch_segmentation_level{level_}_diameter{diameter_}.png", dpi=600)

    return 0


def build_results_dir():
    global RESULTS
    RESULTS = Path("../../scratch/lbrunsch/results/nucleus_segmentation/stardist")
    os.makedirs(RESULTS, exist_ok=True)
    global RESULTS_3D
    RESULTS_3D = RESULTS / "3d_seg"
    os.makedirs(RESULTS_3D, exist_ok=True)


if __name__ == "__main__":

    run = "2D"
    square_size_ = 2000

    build_results_dir()
    init_logger()

    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    if run == "2D":
        StarDist2D.from_pretrained()
        model_type = "2D_versatile_fluo"  # alternative: 2D_versatile_fluo, 2D_paper_dsb2018, 2D_versatile_he, 2D_demo
        image_type = "focus"  # alternative: focus, mip
        optimize_stardist_2d(path_replicate_1, model_type, image_type, square_size=square_size_)
        # Visually inspect optimize stardist result to decide parameters
        prob_threshold = None
        nms_threshold = None
        run_stardist_2d(path_replicate_1, model_type, image_type,
                        prob_thrsh=prob_threshold, nms_thrsh=nms_threshold, square_size=square_size_)
    elif run == "3D":
        level = 0
        StarDist3D.from_pretrained()
        model_type = "3D_demo"
        run_stardist_3d(path_replicate_1, level_=level, model_type_=model_type)

    elif run == "patch":
        StarDist2D.from_pretrained()
        model_type = "2D_versatile_fluo"  # alternative: 2D_versatile_fluo, 2D_paper_dsb2018, 2D_versatile_he, 2D_demo
        image_type = "mip"  # alternative: focus, mip
        run_patch_stardist_2d(path_replicate_1, model_type, image_type)
