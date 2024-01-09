# Std
import os
import pickle
from pathlib import Path
from typing import Optional

# Third party
import matplotlib.pyplot as plt
import numpy as np
import logging
import platform
from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize
from itertools import product

from .. import utils as src_utils
from . import utils as segmentation_utils


if platform.system() != "Windows":
    import resource
    # Set the maximum memory usage in bytes (300GB) = limits of memory resources from RCP cluster
    max_memory = int(3e11)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))

RESULTS = Path()
RESULTS_3D = Path()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = True


def segment_stardist(
        img: np.ndarray,
        model_type_: str,
        do_3d: bool = False,
        prob_thrsh: float = None,
        nms_thrsh: float = None,
        **kwargs
):
    """Run cellpose and get masks

    Parameters
    ----------
    img: Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type_: pretrained models 2D_versatile_fluo / 2D_paper_dsb2018, 2D_versatile_he
    do_3d: perform 3D nuclear segmentation requires 3D array
    nms_thrsh: parameter non-maxima suppression threshold
    prob_thrsh: parameter probability threshold

    Returns
    -------
    np.ndarray
        labelled image, where 0 = no masks; 1, 2, ... = mask labels
    """
    if kwargs:
        raise ValueError("Kwargs should be empty, unexpected parameters:", kwargs)

    if not do_3d:
        model = StarDist2D.from_pretrained(model_type_)
        # normalizer (perform normalization), sparse (aggregation),
        # prob_thresh, nms_thresh (non-maximum suppression), scale (factor), n_tiles (broken up in overlay)
        labels, details = model.predict_instances(normalize(img), prob_thresh=prob_thrsh, nms_thresh=nms_thrsh)
        coord = details["coord"]
    else:
        model = StarDist3D.from_pretrained(model_type_)
        labels, details = model.predict_instances(normalize(img))
        coord = details["points"]

    return build_stardist_mask_outlines(coord)


def build_stardist_mask_outlines(masks):
    masks_outlines = []
    for mask in masks:
        mask = mask.astype(int)
        mask = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
        tmp_1 = mask[0, :].copy()
        tmp_2 = mask[1, :].copy()
        mask[0, :] = tmp_2
        mask[1, :] = tmp_1
        masks_outlines.append(mask)

    return masks_outlines


def optimize_stardist_2d(path_replicate_: Path, model_type_: str, image_type_: str, square_size: Optional[int],
                         level_: int, save_masks: bool = True):

    img = src_utils.load_image(path_replicate_, img_type=image_type_, level_=level_)
    patch, boundaries = src_utils.image_patch(img, square_size_=square_size)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 40))
    [ax.axis("off") for ax in axs.ravel()]

    if square_size is not None:
        [ax.imshow(patch) for ax in axs.ravel()]
    else:  # small region to plot
        og = (patch.shape[0]//2 - 400, patch.shape[0]//2 + 400)
        [ax.imshow(patch[og[0]:og[1], og[0]:og[1]]) for ax in axs.ravel()]

    # default 2D prob = 0.479, nms threshold = 0.3
    prob_thresh = [0.3, 0.4, 0.5, 0.7]
    nms_thresh = [0.1, 0.3, 0.5, 0.7]
    comb = product(prob_thresh, nms_thresh)
    for ax, (prob, nms) in zip(axs.ravel(), comb):
        masks_stardist = segment_stardist(patch, model_type_=model_type_, do_3d=False,
                                                  prob_thrsh=prob, nms_thrsh=nms)

        if save_masks:
            masks_dir = RESULTS / "masks"
            os.makedirs(masks_dir, exist_ok=True)

            with open(masks_dir / f"masks_{model_type_}-nms{nms}-prob{prob}"
                                  f"_{image_type_}-{square_size}.pkl", 'wb') as file:
                pickle.dump(masks_stardist, file)

        ax.set_title(f"Prob: {prob}, Nms: {nms}")
        for mask in masks_stardist:
            if square_size is not None or (square_size is None and
                                           ((og[0] < mask[0, :].max() < og[1]) or
                                            (og[0] < mask[0, :].min() < og[1]) or
                                            (og[0] < mask[1, :].max() < og[1]) or
                                            (og[0] < mask[1, :].min() < og[1]))):

                ax.plot(mask[0, :], mask[1, :], 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(RESULTS / f"stardist_2d_optimization_{image_type_}_{model_type_}_{square_size}.png")
    plt.close()


def run_patch_stardist_2d(path_replicate_: Path, model_type_: str, image_type_: str,
                          nms_thresh: Optional[float], prob_thresh: Optional[float]):

    model_args = {"nms_thrsh": nms_thresh, "prob_thrsh": prob_thresh, "model_type_": model_type_}
    segmentation_type = "stardist"

    segmentation_utils.run_patch_segmentation_2d(path_replicate_, image_type_, segmentation_type, model_args, RESULTS,
                              segment_stardist)


def run_stardist_2d(path_replicate_: Path, model_type_: str, image_type_: str, level_: int = 0,
                    prob_thrsh: Optional[float] = None, nms_thrsh: Optional[float] = None,
                    square_size: Optional[int] = 400):

    model_args = {"nms_thrsh": nms_thrsh, "prob_thrsh": prob_thrsh, "model_type_": model_type_}
    segmentation_type = "stardist"
    segmentation_utils.run_segmentation_2d(path_replicate_, segmentation_type, image_type_, model_args, segment_stardist, level_,
                        square_size, RESULTS)


def run_stardist_location_2d(path_replicate_: Path, model_type_: str, image_type_: str,
                             prob_thrsh: Optional[float] = None, nms_thrsh: Optional[float] = None,
                             square_size: Optional[int] = 400):

    model_args = {"nms_thrsh": nms_thrsh, "prob_thrsh": prob_thrsh, "model_type_": model_type_}
    segmentation_type = "stardist"
    segmentation_utils.run_segmentation_location_2d(path_replicate_, segmentation_type, image_type_, segment_stardist,
                                 square_size, model_args, RESULTS)


def run_stardist_3d(path_replicate_: Path, model_type_: str, level_: int = 0, diameter_: int = 10):

    img = src_utils.load_image(path_replicate_, img_type="stack", level_=level_)
    patch, boundaries = src_utils.image_patch(img, square_size_=700)

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
    RESULTS = Path("../../../scratch/lbrunsch/results/nucleus_segmentation/stardist")
    os.makedirs(RESULTS, exist_ok=True)
    global RESULTS_3D
    RESULTS_3D = RESULTS / "3d_seg"
    os.makedirs(RESULTS_3D, exist_ok=True)


if __name__ == "__main__":

    # Various set up
    src_utils.check_cuda()
    build_results_dir()
    init_logger()

    # Run Parameters
    run = "2D"  # alternative: 3D or Patch
    square_size_ = None
    optimize = True

    # Path
    data_path = Path("../../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    if run == "2D":
        StarDist2D.from_pretrained()

        # Model Version
        model_type = "2D_versatile_fluo"  # alternative: 2D_versatile_fluo, 2D_paper_dsb2018, 2D_versatile_he, 2D_demo

        # Image Type
        image_type = "mip"  # alternative: focus, mip
        level = 0

        if optimize:
            optimize_stardist_2d(path_replicate_1, model_type, image_type, square_size=square_size_, level_=level)

        else:
            # Model Parameters
            prob_threshold = None
            nms_threshold = None

            # Run Stardist at one location and compare to Xenium
            run_stardist_2d(path_replicate_1, model_type, image_type,
                            prob_thrsh=prob_threshold, nms_thrsh=nms_threshold, square_size=square_size_)

            # Run Stardist at various locations
            run_stardist_location_2d(path_replicate_1, model_type, image_type,
                                     prob_thrsh=prob_threshold, nms_thrsh=nms_threshold, square_size=square_size_)

    # Run stardist with various patch size
    elif run == "patch":
        StarDist2D.from_pretrained()

        # Image Type
        image_type = "mip"  # alternative: focus, mip

        # Model Parameters
        model_type = "2D_versatile_fluo"  # alternative: 2D_versatile_fluo, 2D_paper_dsb2018, 2D_versatile_he, 2D_demo
        prob_threshold = None
        nms_threshold = None

        run_patch_stardist_2d(path_replicate_1, model_type, image_type, nms_threshold, prob_threshold)

    # Run Stardist in 3D (needs some work)
    elif run == "3D":
        level = 0
        StarDist3D.from_pretrained()
        model_type = "3D_demo"
        run_stardist_3d(path_replicate_1, level_=level, model_type_=model_type)
