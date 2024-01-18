from pathlib import Path
from typing import Callable
import platform
import pickle

import matplotlib.pyplot as plt


if platform.system() == "Linux":
    from .. import utils as src_utils

    WORKING_DIR = Path("..")
else:
    import src.utils as src_utils

    WORKING_DIR = Path("../../..")

def get_xenium_nucleus_boundaries(path_replicate_: Path, boundaries: list = None):

    adata = src_utils.load_xenium_data(Path(str(path_replicate_) + ".h5ad"))

    # Convert xenium predefined nucleus boundaries to pixels locations
    # (x,y): vertex_x is the horizontal axis / vertex y is the vertical axis
    # from_metadata 1 pixel = 0.2125 microns
    x_conversion = 0.2125
    y_conversion = 0.2125
    adata.uns["nucleus_boundaries"]["vertex_y_pixel"] = adata.uns["nucleus_boundaries"]["vertex_y"].apply(
        lambda p: round(p/y_conversion))
    adata.uns["nucleus_boundaries"]["vertex_x_pixel"] = adata.uns["nucleus_boundaries"]["vertex_x"].apply(
        lambda p: round(p/x_conversion))

    if boundaries is None:
        boundaries = [[0, max(adata.uns["nucleus_boundaries"]["vertex_y"])],
                      [0, max(adata.uns["nucleus_boundaries"]["vertex_x"])]]

    # Selection of segmented nucleus that are inside the patch
    pix_boundaries = adata.uns["nucleus_boundaries"][(adata.uns["nucleus_boundaries"]["vertex_x_pixel"] > boundaries[1][0]) &
                                                     (adata.uns["nucleus_boundaries"]["vertex_x_pixel"] < boundaries[1][1]) &
                                                     (adata.uns["nucleus_boundaries"]["vertex_y_pixel"] > boundaries[0][0]) &
                                                     (adata.uns["nucleus_boundaries"]["vertex_y_pixel"] < boundaries[0][1])
                                                     ]
    output = [[], []]
    for cell_seg in pix_boundaries["cell_id"].unique():
        output[0].append(pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy() - boundaries[1][0])
        output[1].append(pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy() - boundaries[0][0])

    return output


def get_masks(method: str, params: dict, img_type_: str, square_size_: int = None):

    masks_dir = WORKING_DIR / "scratch/lbrunsch/results/nucleus_segmentation"

    if method == "cellpose":
        try:
            masks_dir = masks_dir / "cellpose/masks"
            with open(masks_dir / f"masks_{params['model']}-diameter{params['diameter_']}"
                                  f"_{img_type_}-{square_size_}.pkl", 'rb') as file:
                masks_cellpose = pickle.load(file)
        except Exception as e:
            print(f"Error: {e}")
            print(f"CellPose masks with: {params}, image type-{img_type_} and square_size-{square_size_} does not"
                  f"exist")



def run_segmentation_2d(path_replicate_: Path, model_type_: str, image_type_: str, model_args: dict, segment_fct: Callable,
                        level_: int = 0, square_size: int = 400, results_dir: Path = Path()):

    img = src_utils.load_image(path_replicate_, img_type=image_type_, level_=level_)
    patch, boundaries = src_utils.image_patch(img, square_size_=square_size)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    [ax.axis("off") for ax in axs]
    [ax.imshow(patch) for ax in axs]

    masks = segment_fct(patch, do_3d=False, **model_args)

    model_version = ""
    if model_type_ == "stardist":
        model_version = model_args["model_type_"]
        axs[2].set_title(f"Stardist - {model_version}, nms thresh, prob thresh")
    elif model_type_ == "cellpose":
        model_version = model_args["model_type"]
        axs[2].set_title(f"CellPose - {model_version}")
    elif model_type_ == "watershed":
        axs[2].set_title(f"Watershed")
    else:
        raise ValueError("unknown")

    axs[0].set_title("Original Image")
    for mask in masks:
        axs[2].plot(mask[0, :], mask[1, :], 'r', linewidth=.8)

    # Get boundaries from Xenium
    masks_xenium = get_xenium_nucleus_boundaries(path_replicate_, boundaries)
    axs[1].set_title("Xenium Segmentation")
    for x, y in zip(masks_xenium[0], masks_xenium[1]):
        axs[1].plot(x, y, 'r', linewidth=.8)

    plt.savefig(results_dir / f"{model_type_}-{model_version}_{image_type_}-{square_size}.png")
    plt.close()


def run_patch_segmentation_2d(path_replicate_: Path, image_type_: str, model_type_: str, model_args: dict,
                              results_dir: Path, segment_fct: Callable):

    img = src_utils.load_image(path_replicate_, img_type=image_type_)

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 30))
    [ax.axis("off") for ax in axs.ravel()]

    model_version = ""
    initial = 400
    square_sizes = [400, 600, 800, 1000, 1600, 2000, 4000, 8000]

    for ax, size in zip(axs.ravel(), square_sizes):

        patch, boundaries = src_utils.image_patch(img, square_size_=size)
        ax.imshow(patch)

        masks = segment_fct(patch, do_3d=False, **model_args)

        if model_type_ == "stardist":
            model_version = model_args["model_type_"]
            fig.suptitle(f"Stardist - {model_version} - nms thresh - prob thresh")
        elif model_type_ == "cellpose":
            model_version = model_args["model_type"]
            fig.suptitle(f"Cellpose - {model_version}")
        elif model_type_ == "watershed":
            fig.suptitle(f"Watershed Algorithm")
        else:
            raise ValueError("unknown")

        ax.set_title(f"Square Size = {size}")
        center_x = patch.shape[0] // 2
        min_ = center_x - initial // 2
        max_ = center_x + initial // 2

        for mask in masks:
            # check if the mask is inside the patch
            if (((max_ > mask[1, :].max() > min_) or (max_ > mask[1, :].min() > min_))
               and ((max_ > mask[0, :].max() > min_) or (max_ > mask[0, :].min() > min_))):

                mask[0, :] = mask[0, :] - (patch.shape[0] // 2 - initial // 2)
                mask[1, :] = mask[1, :] - (patch.shape[1] // 2 - initial // 2)
                ax.plot(mask[0, :], mask[1, :], 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(results_dir / f"{model_type_}-{model_version}_{image_type_}-{square_sizes}"
                              f"2d_patches_patches.png")
    plt.close()


def run_segmentation_location_2d(path_replicate_: Path, segment_type_: str, image_type_: str, segment_fct: Callable,
                                 square_size: int = 400, model_args: dict = {}, results_dir: Path = Path()):

    img = src_utils.load_image(path_replicate_, img_type=image_type_, level_=0)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    [ax.axis("off") for ax in axs.ravel()]

    # Image (mip and focus): [33'0000 x 48'0000] pixels at level_0
    square_origin = [(15000, 15000), (30000, 30000), (10000, 10000), (20000, 1000),
                     (15000, 2000), (2000, 19000), (5000, 5000), (5000, 30000), (5000, 15000)]

    model_version = ""
    if segment_type_ == "stardist":
        model_version = model_args["model_type_"]
    elif segment_type_ == "cellpose":
        model_version = model_args["model_type"]

    for i, ax in enumerate(axs.ravel()):
        patch, boundaries = src_utils.image_patch(img, square_size_=square_size, orig_=square_origin[i])
        ax.imshow(patch)
        ax.set_title(f"Patch ({square_origin[i][0]},{square_origin[i][1]})")
        masks = segment_fct(patch, do_3d=False, **model_args)

        for mask in masks:
            ax.plot(mask[0, :], mask[1, :], 'r', linewidth=.8)

    plt.savefig(results_dir / f"{segment_type_}-{model_version}_{image_type_}-{square_size}_locations.png")
    plt.close()
