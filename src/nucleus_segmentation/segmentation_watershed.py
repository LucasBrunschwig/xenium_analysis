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
from cellpose.utils import outlines_list
import squidpy as sq
from squidpy.im import ImageContainer

# Relative import
from src.utils import load_image, image_patch, load_xenium_data


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


def segment_watershed(
        img: np.ndarray,
        do_3d: bool = False,
        **kwargs,
):
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

    img = ImageContainer(img=img, layer="image")

    if not do_3d:
        sq.im.process(img=img, layer="image", method="smooth", sigma=[2, 2, 0, 0],
                      )

        # Watershed is a method developed by CellProfiler
        sq.im.segment(img=img, layer="image_smooth", method="watershed", channel_ids=0, chunks=1000)
    else:
        pass

    return build_watershed_mask_outlines(img.data.segmented_watershed.data[:, :, 0, 0])


def run_patch_watershed_2d(path_replicate_: Path, image_type_: str, level_: int = 0):

    img = load_image(path_replicate_, img_type=image_type_, level_=level_)
    initial = 400
    square_sizes = [400, 600, 800, 1000, 1600, 2000, 4000, 8000]

    patch, _ = image_patch(img, square_size=square_sizes[0])

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 30))
    [ax.axis("off") for ax in axs.ravel()]
    [ax.imshow(patch) for ax in axs.ravel()]

    for ax, size in zip(axs.ravel(), square_sizes):
        patch, boundaries = image_patch(img, square_size_=size)

        labels, masks_stardist = segment_watershed(patch, do_3d=False)

        ax.set_title(f"Square Size = {size}")
        center_x = patch.shape[0] // 2
        min_ = center_x - initial // 2
        max_ = center_x + initial // 2

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
    plt.savefig(RESULTS / f"watershed_2d_{image_type_}_patches.png")
    plt.close()


def build_watershed_mask_outlines(masks):
    mask_outlines = []
    masks = outlines_list(masks, multiprocessing=False)
    for mask in masks:
        if mask.shape[0] > 0:
            mask = np.array(mask).T
            mask_outlines.append(np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1))

    return mask_outlines


def run_watershed_2d(path_replicate_: Path, image_type_: str, level_: int = 0, square_size: int = 400):

    img = load_image(path_replicate_, img_type=image_type_, level_=level_)
    patch, boundaries = image_patch(img, square_size_=square_size)

    masks_watershed = segment_watershed(patch, do_3d=False)

    masks_xenium = get_xenium_nucleus_boundaries(path_replicate_, boundaries)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    [ax.axis("off") for ax in axs]
    [ax.imshow(patch) for ax in axs]
    axs[0].set_title("Original Image")
    axs[2].set_title("Watershed Segmentation")
    for mask in masks_watershed:
        mask = np.array(mask).T
        closed_contour = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
        axs[2].plot(closed_contour[0, :], closed_contour[1, :], 'r', linewidth=.8)

    axs[1].set_title("Xenium Segmentation")
    for x, y in zip(masks_xenium[0], masks_xenium[1]):
        axs[1].plot(x, y, 'r', linewidth=.8)

    plt.savefig(RESULTS / f"watershed_2d_{image_type_}_{square_size}.png")
    plt.close()


def run_stardist_location_2d(path_replicate_: Path, image_type_: str, level_: int = 0,
                             square_size: int = 400):

    img = load_image(path_replicate_, img_type=image_type_, level_=level_)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    [ax.axis("off") for ax in axs.ravel()]

    # Image (mip and focus): [33'0000 x 48'0000] pixels
    square_origin = [(15000, 15000), (30000, 30000), (10000, 10000), (20000, 1000),
                     (15000, 2000), (2000, 19000), (5000, 5000), (5000, 30000), (5000, 15000)]

    for i, ax in enumerate(axs.ravel()):
        patch, boundaries = image_patch(img, square_size=square_size, orig=square_origin[i])
        ax.imshow(patch)
        ax.set_title(f"Patch ({square_origin[i][0]},{square_origin[i][1]})")
        masks = segment_watershed(patch, do_3d=False)

        for mask in masks:
            if mask.shape[0] > 0 and mask.shape[1] > 0 :
                mask = np.array(mask).T
                closed_contour = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
                ax.plot(closed_contour[0, :], closed_contour[1, :], 'r', linewidth=.8)

    plt.savefig(RESULTS / f"watershed_2d_{image_type_}_locations.png")
    plt.close()


# def run_stardist_3d(path_replicate_: Path, model_type_: str, level_: int = 0, diameter_: int = 10):
#
#     img = load_image(path_replicate_, img_type="stack", level_=level_)
#     patch, boundaries = image_patch(img, square_size=700)
#
#     fig, axs = plt.subplots(3, 4)
#
#     for i, (layer, ax) in enumerate(zip(patch, axs.ravel())):
#         ax.axis("off")
#         ax.set_title(f"patch - layer {i}")
#         ax.imshow(layer)
#
#     plt.tight_layout()
#     fig.savefig(RESULTS_3D / f"3d_patch_og_level{level_}_diameter{diameter_}.png", dpi=600)
#
#     print("Segmenting the whole image")
#
#     if not os.path.isfile(RESULTS_3D / f"stardist_mask_level{level_}.pkl"):
#         seg_3d, coord = segment_stardist(patch, model_type_=model_type_, do_3d=True)
#         with open(RESULTS_3D / f"stardist_mask_level{level_}.pkl", "wb") as file:
#             pickle.dump(seg_3d, file)
#     else:
#         with open(RESULTS_3D / f"stardist_mask_level{level_}.pkl", "rb") as file:
#             seg_3d = pickle.load(file)
#
#     print("Plotting Resulting Segmentation")
#
#     fig, axs = plt.subplots(3, 4)
#
#     for i, (layer, ax) in enumerate(zip(patch, axs.ravel())):
#         ax.axis("off")
#         ax.set_title(f"stardist - layer {i}")
#         ax.imshow(layer)
#         ax.imshow(seg_3d[i, :, :])
#
#     plt.tight_layout()
#     fig.savefig(RESULTS_3D / f"3d_patch_segmentation_level{level_}_diameter{diameter_}.png", dpi=600)
#
#     return 0


def build_results_dir():
    global RESULTS
    RESULTS = Path("../../scratch/lbrunsch/results/nucleus_segmentation/watershed")
    os.makedirs(RESULTS, exist_ok=True)
    global RESULTS_3D
    RESULTS_3D = RESULTS / "3d_seg"
    os.makedirs(RESULTS_3D, exist_ok=True)


if __name__ == "__main__":

    run = "2D"
    square_size_ = 500

    build_results_dir()
    init_logger()

    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    if run == "2D":
        image_type = "mip"  # alternative: focus, mip
        run_watershed_2d(path_replicate_1, image_type, square_size=square_size_)
        run_stardist_location_2d(path_replicate_1, image_type)

    elif run == "patch":
        image_type = "mip"
        run_patch_watershed_2d(path_replicate_1, image_type)
