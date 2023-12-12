# Std
import os
from pathlib import Path

# Third party
from cellpose import models
from tifffile import tifffile
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path()


def segment_cellpose(
        img: np.ndarray,
        model_type: str = "nuclei",
        net_avg: bool = False,
) -> np.ndarray:
    """Run cellpose and get masks

    Parameters
    ----------
    img : Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type: model type to load
    net_avg: runs 1 model or 4 built-in models

    Returns
    -------
    NDArray
        labelled image, where 0=no masks; 1,2,...=mask labels
    """

    # Init model
    model = models.Cellpose(gpu=True, model_type=model_type)

    # Eval model
    # Hyperparameters:
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
    masks, flows, styles, diameters = model.eval(x=img, channels=[0, 0])

    return masks


def image_patch(img_array, square_size: int = 400, format: str = "test"):
    """

    Parameters
    ----------
    img_array
    square_size: the length of the image square
    format: "test" returns one patch starting at 0,0 |
            "random" returns a random location
            "training": returns a lit of patches adapting the square size to match the sequence length

    Returns
    -------

    """

    if format == "test":
        return img_array[img_array.shape[0]//2-square_size//2:img_array.shape[0]//2+square_size//2,
                         img_array.shape[1]//2-square_size//2:img_array.shape[1]//2+square_size//2]
    else:
        raise NotImplementedError(f" {format} not implemented yet")


def run_cellpose(path_replicate: Path, img_type: str = "mip"):
    """ This function run cellpose on an image

    Parameters
    ----------
    path_replicate (Path): path to the replicate
    img_type (str): type of images

    Returns
    -------

    """
    if img_type == "mip":
        img_file = str(path_replicate / "morphology_mip.ome.tif")
    elif img_type == "focus":
        img_file = str(path_replicate / "morphology_focus.ome.tif")
    else:
        raise ValueError("No Values")

    img = tifffile.imread(img_file)

    # Debugging
    debug = True
    if debug:
        patch = image_patch(img, square_size=1000, format="test")
    else:
        patch = img

    seg_patch_nuclei = segment_cellpose(patch, model_type="nuclei")
    seg_patch_cyto = segment_cellpose(patch, model_type="cyto")
    seg_patch_cyto2 = segment_cellpose(patch, model_type="cyto2")

    # Plot the results and compare it to the original images
    fig, ax = plt.subplots(2, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle(f"Nucleus Segmentation for Pretrained Models on {img_type.upper()} morphology")
    [x_.axis("off") for x_ in ax.ravel()]
    ax[0, 1].set_title("Original DAPI Image")
    ax[0, 1].imshow(patch)
    ax[1, 0].imshow(seg_patch_nuclei)
    ax[1, 0].set_title("CellPose - Nucleus")
    ax[1, 1].imshow(seg_patch_cyto)
    ax[1, 1].set_title("CellPose - Cyto")
    ax[1, 2].imshow(seg_patch_cyto2)
    ax[1, 2].set_title("CellPose - Cyto2")
    plt.tight_layout()
    fig.savefig(RESULTS / f"cellpose_{img_type}_segmentation.png", bbox_inches="tight")
    plt.show()

    return fig


def build_results_dir():
    global RESULTS
    RESULTS = Path("../../scratch/lbrunsch/results/nucleus_segmentation")
    os.makedirs(RESULTS, exist_ok=True)


if __name__ == "__main__":

    build_results_dir()

    print("Testing CellPose Segmentation Algorithm on Nuclei")
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    img_type = "focus"
    run_cellpose(path_replicate_1, img_type)
