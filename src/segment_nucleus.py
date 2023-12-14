# Std
import os
from pathlib import Path

# Third party
from cellpose import models
from cellpose.utils import outlines_list
from tifffile import tifffile
import matplotlib.pyplot as plt
import numpy as np

# Relative import
from utils import load_xenium_data

RESULTS = Path()


def segment_cellpose(
        img: np.ndarray,
        model_type: str = "nuclei",
        net_avg: bool = False,
) -> np.ndarray:
    """Run cellpose and get masks

    Parameters
    ----------
    img: Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type: model type to load
    net_avg: runs 1 model or 4 built-in models

    Returns
    -------
    np.ndarray
        labelled image, where 0 = no masks; 1, 2, ... = mask labels
    """

    # Init model
    model = models.Cellpose(gpu=True, model_type=model_type)

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
    masks, flows, styles, diameters = model.eval(x=img, channels=[0, 0], net_avg=net_avg, diameter=20)

    return masks


def image_patch(img_array, square_size: int = 400, format_: str = "test"):
    """

    Parameters
    ----------
    img_array
    square_size: the length of the image square
    format_: "test" returns one square patch at the image center (width = square size)
             "training": returns a list of patches adapting the square size to match the image size

    Returns
    -------
    returns: list of patches or one patch as np.ndarray
    """

    if format_ == "test":
        return [img_array[img_array.shape[0]//2-square_size//2:img_array.shape[0]//2+square_size//2,
                img_array.shape[1]//2-square_size//2:img_array.shape[1]//2+square_size//2],
                ([img_array.shape[0]//2-square_size//2, img_array.shape[0]//2+square_size//2],
                [img_array.shape[1]//2-square_size//2, img_array.shape[1]//2+square_size//2])]
    else:
        raise NotImplementedError(f" {format_} not implemented yet")


def load_image(path_replicate: Path, img_type: str):
    if img_type == "mip":
        img_file = str(path_replicate / "morphology_mip.ome.tif")
    elif img_type == "focus":
        img_file = str(path_replicate / "morphology_focus.ome.tif")
    elif img_type == "stack":
        img_file = str(path_replicate / "morphology.ome.tif")
    else:
        raise ValueError("Not a type of image")

    return tifffile.imread(img_file)


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
    patch, boundaries = image_patch(img, square_size=700, format_="test")

    adata = load_xenium_data(Path(str(path_replicate) + ".h5ad"))

    # Convert xenium predefined nucleus boundaries to pixels locations
    # (x,y): vertex_x is the horizontal axis / vertex y is the vertical axis
    # from_metadata 1 pixel = 0.2125 microns
    x_conversion = 0.2125
    y_conversion = 0.2125
    adata.uns["nucleus_boundaries"]["vertex_y_pixel"] = adata.uns["nucleus_boundaries"]["vertex_y"].apply(
        lambda p: round(p/y_conversion) - boundaries[0][0])
    adata.uns["nucleus_boundaries"]["vertex_x_pixel"] = adata.uns["nucleus_boundaries"]["vertex_x"].apply(
        lambda p: round(p/x_conversion) - boundaries[1][0])

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
    plt.savefig(RESULTS / f"og_patch_{img_type}.png")
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
    for cell_seg in pix_boundaries["cell_id"].unique():
        x = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy()
        y = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy()
        ax[2, 0].plot(x, y, c='r', linewidth=.5)
        ax[2, 1].plot(x, y, c='r', linewidth=.5)
        ax[2, 2].plot(x, y, c='r', linewidth=.5)

    plt.tight_layout()
    fig.savefig(RESULTS / f"cellpose_{img_type}_segmentation.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 3)
    [x_.axis("off") for x_ in ax]
    [x_.imshow(patch) for x_ in ax]
    [ax[0].plot(mask[:, 0], mask[:, 1], 'g', linewidth=.5) for mask in seg_patch_comb_outlines]
    [ax[2].plot(mask[:, 0], mask[:, 1], 'g', linewidth=.5) for mask in seg_patch_comb_outlines]

    for cell_seg in pix_boundaries["cell_id"].unique():
        x = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy()
        y = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy()
        ax[1].plot(x, y, c='r', linewidth=.5)
        ax[2].plot(x, y, c='r', linewidth=.5)
    plt.tight_layout()
    fig.savefig(RESULTS / f"superposition_xenium_cellpose_{img_type}.png")

    return 0


def run_cellpose_3d(path_replicate_: Path, level: int = 0):
    pass


def build_results_dir():
    global RESULTS
    RESULTS = Path("../../scratch/lbrunsch/results/nucleus_segmentation")
    os.makedirs(RESULTS, exist_ok=True)


if __name__ == "__main__":

    run = "2d"

    build_results_dir()

    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    if run == "2d":
        print("Running 2D Segmentation Algorithm on Nuclei")
        img_type = "focus"
        run_cellpose_2d(path_replicate_1, img_type)
    elif run == "3d":
        level = 0
        run_cellpose_3d(path_replicate_1, level)

