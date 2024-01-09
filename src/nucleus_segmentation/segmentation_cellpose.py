# Std
import os
import pickle
from pathlib import Path

# Third party
from cellpose import models
from cellpose.utils import outlines_list
from cellpose.contrib.distributed_segmentation import segment
from tifffile import tifffile
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging

# Relative import
from src.utils import load_xenium_data, load_image, image_patch

import platform


if platform.system() != "Windows":
    import resource
    # Set the maximum memory usage in bytes (300GB)
    max_memory = int(3e11)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))

RESULTS = Path()
RESULTS_3D = Path()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = True


def segment_cellpose(
        img: np.ndarray,
        model_type: str = "nuclei",
        net_avg: bool = False,
        do_3d: bool = False,
        diameter: int = 30,
        **kwargs
) -> np.ndarray:
    """Run cellpose and get masks

    Parameters
    ----------
    img: Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type: model type to load
    net_avg: evaluate 1 model or average of 4 built-in models
    do_3d: perform 3D nuclear segmentation requires 3D array
    diameter: estimated size of nucleus

    Returns
    -------
    np.ndarray
        labelled image, where 0 = no masks; 1, 2, ... = mask labels
    """

    # Init model
    model = models.Cellpose(gpu=True, model_type=model_type)

    if do_3d:
        # fast_mode, use_anisotropy, iou_depth, iou_threshold
        masks = segment(img, channels=[0, 0], model_type=model_type, diameter=diameter)

    else:
        # Eval model
        # Various Argument:
        # - x: list of array of images list(2D/3D) or array of 2D/3D images, or 4D array of image
        # - channels: length(2)
        #       - 1: channel to segment (0=grayscale, 1=red, 2=green, 3=blue)
        #       - 2: optional nuclear channel (0=none, 1=red, 2=green 3=blue)
        #       in DAPI images no different channels for nucleus
        # - invert(false), normalize(true)
        # - net_avg: 4 built-in networks and averages them (false)
        # - diameter (default: 30), flow threshold (0.4)
        # - batch size (224x224 patches to run simultaneously
        # - augment/tile/tile_overlap/resample/interp/cellprob_threshold/min_size/stitch_threshold
        masks, flows, styles, diameters = model.eval(x=[img], batch_size=16, channels=[0, 0], net_avg=net_avg,
                                                     diameter=diameter, do_3D=do_3d, progress=True)

    return build_cellpose_mask_outlines(masks)


def build_cellpose_mask_outlines(masks):
    return outlines_list(masks[0], multiprocessing=False)


def run_cellpose_2d(path_replicate: Path, img_type: str = "mip"):
    """ This function run cellpose on an image

    Parameters
    ----------
    path_replicate (Path): path to the replicate
    img_type (str): type of images

    Returns
    -------

    """

    # Load Image Type
    img = load_image(path_replicate, img_type)

    # Returns a test patch with the image boundaries
    patch, boundaries = image_patch(img, square_size_=700, format_="test")

    adata = load_xenium_data(Path(str(path_replicate) + ".h5ad"))

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

    # Run CellPose with 3 predefined models (nuclei, cyto, cyto2)
    seg_patch_nuclei = segment_cellpose(patch, model_type="nuclei")
    seg_patch_nuclei_outlines = outlines_list(seg_patch_nuclei, multiprocessing=False)

    seg_patch_cyto = segment_cellpose(patch, model_type="cyto")
    seg_patch_cyto_outlines = outlines_list(seg_patch_cyto, multiprocessing=False)

    seg_patch_cyto2 = segment_cellpose(patch, model_type="cyto2")
    seg_patch_cyto2_outlines = outlines_list(seg_patch_cyto2, multiprocessing=False)

    seg_patch_comb = segment_cellpose(patch, net_avg=True)
    seg_patch_comb_outlines = outlines_list(seg_patch_comb, multiprocessing=False)

    plt.imshow(patch)
    plt.savefig(RESULTS / f"og_patch_{img_type}.png", dpi=500)
    plt.close()

    # Plot the results and compare it to the original images
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle(f"Nucleus Segmentation for Pretrained Models on {img_type.upper()} morphology")
    [x_.axis("off") for x_ in ax.ravel()]
    [x_.imshow(patch) for x_ in ax.ravel()]
    ax[0, 1].set_title("Original DAPI Image")
    [ax[1, 0].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5) for mask in seg_patch_nuclei_outlines]
    ax[1, 0].set_title("CellPose - Nucleus")
    [ax[1, 1].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5) for mask in seg_patch_cyto_outlines]
    ax[1, 1].set_title("CellPose - Cyto")
    [ax[1, 2].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5) for mask in seg_patch_cyto2_outlines]
    ax[1, 2].set_title("CellPose - Cyto2")

    # Plot Xenium original boundaries
    [ax[2, i].set_title("Xeinum Segmentation") for i in [0, 1, 2]]
    for cell_seg in pix_boundaries["cell_id"].unique():
        x = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy() - boundaries[1][0]
        y = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy() - boundaries[0][0]
        ax[2, 0].plot(x, y, c='r', linewidth=.5)
        ax[2, 1].plot(x, y, c='r', linewidth=.5)
        ax[2, 2].plot(x, y, c='r', linewidth=.5)

    plt.tight_layout()
    fig.savefig(RESULTS / f"cellpose_{img_type}_segmentation.png", bbox_inches="tight", dpi=500)

    fig, ax = plt.subplots(1, 3)
    [x_.axis("off") for x_ in ax]
    [x_.imshow(patch) for x_ in ax]
    ax[0].set_title("CellPose - Average")
    ax[1].set_title("Xenium Segmentation")
    ax[2].set_title("Segmentation Superposition")
    for cell_seg in pix_boundaries["cell_id"].unique():
        x = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy() - boundaries[1][0]
        y = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy() - boundaries[0][0]
        ax[1].plot(x, y, c='r', linewidth=.5)
        ax[2].plot(x, y, c='r', linewidth=.5)
    [ax[0].plot(mask[:, 0], mask[:, 1], 'aqua', linewidth=.5, alpha=0.5) for mask in seg_patch_comb_outlines]
    [ax[2].plot(mask[:, 0], mask[:, 1], 'aqua', linewidth=.5, alpha=0.5) for mask in seg_patch_comb_outlines]
    plt.tight_layout()
    fig.savefig(RESULTS / f"superposition_xenium_cellpose_{img_type}.png", dpi=500)

    return 0


def run_cellpose_3d(path_replicate_: Path, level_: int = 0, diameter_: int = 10):

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

    if not os.path.isfile(RESULTS_3D / "mask_outline.pkl"):
        seg_3d = segment_cellpose(img, do_3d=True, diameter=diameter_)
        with open(RESULTS_3D / f"mask_level{level_}_diameter{diameter_}.pkl", "wb") as file:
            pickle.dump(seg_3d, file)
        seg_3d_outlines = outlines_list(seg_3d, multiprocessing=False)
        with open(RESULTS_3D / f"mask_outline{level_}_diameter{diameter_}.pkl", "wb") as file:
            pickle.dump(seg_3d_outlines, file)
    else:
        with open(RESULTS_3D / f"mask_outline{level_}_diameter{diameter_}.pkl", "rb") as file:
            seg_3d_outlines = pickle.load(file)

    print("Plotting Resulting Segmentation")

    seg_3d_outlines = seg_3d_outlines[boundaries[1][0]:boundaries[1][1],
                                      boundaries[2][0]:boundaries[2][1]]

    fig, axs = plt.subplots(3, 4)

    for i, (layer, ax) in enumerate(zip(patch, axs.ravel())):
        ax.axis("off")
        ax.set_title(f"nucleus segmentation - layer {i}")
        ax.imshow(layer)
        [ax[i].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5, alpha=1) for mask in seg_3d_outlines[i, :, :]]

    plt.tight_layout()
    fig.savefig(RESULTS_3D / f"3d_patch_segmentation_level{level_}_diameter{diameter_}.png", dpi=600)

    return 0


def transcripts_assignments(masks: np.ndarray, adata, save_path: str, qv_cutoff: float = 20.0):
    """ This is a function that will be moved in another section but used testing nucleus transcripts assignments """

    transcripts_df = adata.uns["spots"]

    mask_dims = {"z_size": masks.shape[0], "x_size": masks.shape[2], "y_size": masks.shape[1]}

    # Iterate through all transcripts
    transcripts_nucleus_index = []
    for index, row in transcripts_df.iterrows():

        x = row['x_location']
        y = row['y_location']
        z = row['z_location']
        qv = row['qv']

        # Ignore transcript below user-specified cutoff
        if qv < qv_cutoff:
            continue

        # Convert transcript locations from physical space to image space
        pix_size = 0.2125
        z_slice_micron = 3
        x_pixel = x / pix_size
        y_pixel = y / pix_size
        z_slice = z / z_slice_micron

        # Add guard rails to make sure lookup falls within image boundaries.
        x_pixel = min(max(0, x_pixel), mask_dims["x_size"] - 1)
        y_pixel = min(max(0, y_pixel), mask_dims["y_size"] - 1)
        z_slice = min(max(0, z_slice), mask_dims["z_size"] - 1)

        # Look up cell_id assigned by Cellpose. Array is in ZYX order.
        nucleus_id = masks[round(z_slice), round(y_pixel), round(x_pixel)]

        # If cell_id is not 0 at this point, it means the transcript is associated with a cell
        if nucleus_id != 0:
            # Increment count in feature-cell matrix
            transcripts_nucleus_index.append(nucleus_id)
        else:
            transcripts_nucleus_index.append(None)

    adata.uns["spot"]["nucleus_id"] = transcripts_nucleus_index
    if str(save_path).endswith("h5ad"):
        adata.write_h5ad(save_path)
    else:
        adata.write_h5ad(str(save_path)+".h5ad")

    return adata


def build_results_dir():
    global RESULTS
    RESULTS = Path("../../scratch/lbrunsch/results/nucleus_segmentation/cellpose")
    os.makedirs(RESULTS, exist_ok=True)
    global RESULTS_3D
    RESULTS_3D = RESULTS / "3d_segmentation"
    os.makedirs(RESULTS_3D, exist_ok=True)


if __name__ == "__main__":

    run = "3d"

    build_results_dir()
    init_logger()

    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    if run == "2d":
        print("Running 2D Segmentation Algorithm on Nuclei")
        img_type = "mip"
        run_cellpose_2d(path_replicate_1, img_type)
    elif run == "3d":
        level = 4
        run_cellpose_3d(path_replicate_1, level, diameter_=10)

