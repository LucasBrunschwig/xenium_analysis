"""
In this image

Implementations:
[  ]:


"""


# Std
import os
import pickle
from pathlib import Path
from typing import Optional

# Third party
import matplotlib.pyplot as plt
import numpy as np
import logging
from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize

import src.nucleus_segmentation.utils as segmentation_utils
import src.utils as src_utils


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = True


def segment_stardist(
        img: np.ndarray,
        model_type_: str,
        n_tiles: tuple,
        do_3d: bool = False,
        prob_thrsh: float = None,
        nms_thrsh: float = None,
        scale: float = 0.2125/0.5,
        pmin: float = 1.,
        pmax: float = 99.,
):
    """Run cellpose and get masks

    Parameters
    ----------
    img: Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type_: pretrained models 2D_versatile_fluo / 2D_paper_dsb2018, 2D_versatile_he
    n_tiles: tuple representing the number of tiles
    do_3d: perform 3D nuclear segmentation requires 3D array
    nms_thrsh: parameter non-maxima suppression threshold
    prob_thrsh: parameter probability threshold
    scale: the scaling to match the resolution (stardist was trained with resolution 0.5 um / pixels
    pmin: normalization parameters

    Returns
    -------
    np.ndarray
        labelled image, where 0 = no masks; 1, 2, ... = mask labels
    """

    if len(img.shape) == 3:
        n_tiles = (n_tiles[0], n_tiles[1], 1)

    if not do_3d:
        model = StarDist2D.from_pretrained(model_type_)
        # normalizer (perform normalization), sparse (aggregation),
        # prob_thresh, nms_thresh (non-maximum suppression), scale (factor), n_tiles (broken up in overlay)
        img_normalized = normalize(img, pmin, pmax, axis=(0, 1))
        labels, details = model.predict_instances(img_normalized, prob_thresh=prob_thrsh, nms_thresh=nms_thrsh,
                                                  n_tiles=n_tiles, scale=scale)
        coord = details["coord"]
    else:
        model = StarDist3D.from_pretrained(model_type_)
        labels, details = model.predict_instances(normalize(img))
        coord = details["points"]

    return build_stardist_mask_outlines(coord), details


def build_stardist_mask_outlines(masks):
    masks_outlines = []
    for mask in masks:
        mask = mask.astype(int)
        mask = np.concatenate((mask, mask[:, 0].reshape((2, 1))), axis=1)
        masks_outlines.append(mask)

    return masks_outlines


def optimize_stardist_2d(img: np.ndarray, img_type: str, model_type_: str, square_size: Optional[int],
                         compute_masks: bool = True, save_path: Path = Path(), pmin: float = 1., pmax:float = 99., scale: float = 0.2125/0.4):

    patch, boundaries = src_utils.image_patch(img, square_size_=square_size, type_=img_type)

    # default 2D prob = 0.479, nms threshold = 0.3
    prob_thresh = [0.3, 0.4, 0.5, 0.7]
    nms_thresh = [0.1, 0.3, 0.5, 0.7]
    masks_dir = save_path / "masks"

    if compute_masks:
        for i, nms in enumerate(nms_thresh):

            masks_stardist, details = segment_stardist(patch, model_type_=model_type_, do_3d=False,
                                                       prob_thrsh=min(prob_thresh), nms_thrsh=nms, n_tiles=(5, 5), pmin=pmin, pmax=pmax, scale=scale)
            for j, prob in enumerate(prob_thresh):

                if prob != min(prob_thresh):
                    masks_stardist = masks_stardist[0:len(np.where(details["prob"] > prob)[0])]

                os.makedirs(masks_dir, exist_ok=True)

                with open(masks_dir / f"masks_{model_type_}-nms{nms}-prob{prob}_p{pmin}-p{pmax}"
                                      f"_scale{scale}_{image_type}-{square_size}.pkl", 'wb') as file:
                    pickle.dump(masks_stardist, file)

    masks_path = []
    subplot_labels = []
    for nms in nms_thresh:
        for prob in prob_thresh:
            path_ = masks_dir / f"masks_{model_type_}-nms{nms}-prob{prob}_p{pmin}-p{pmax}_scale{scale}_{image_type}-{square_size}.pkl"
            masks_path.append(path_)
            subplot_labels.append(f"nms - {nms}, prob {prob}")

    if square_size is None:

        locations = [(15000, 15000), (20000, 30000), (10000, 10000), (22400, 3830), (21080, 20900), (2000, 19000),
                     (5000, 5000), (3850, 22600), (5000, 15000)]

        img_save_path = save_path / img_type / f"pmin{pmin}-pmax{pmax}"
        os.makedirs(img_save_path, exist_ok=True)

        for location in locations:
            visualize_model_params(img, masks_path, location, square_size=400, n_cols=4, n_rows=4, save_path=img_save_path,
                                   subplot_labels=subplot_labels, sub_image=True)

    else:
        img_save_path = save_path / img_type / f"pmin{pmin}-pmax{pmax}"
        os.makedirs(img_save_path, exist_ok=True)
        visualize_model_params(img, masks_path, [img.shape[0] // 2, img.shape[1] // 2], square_size, n_cols=4,
                               n_rows=4, save_path=img_save_path, subplot_labels=subplot_labels, sub_image=False)


def visualize_model_params(img, masks_path: list, location: tuple, square_size: int, n_cols: int, n_rows: int,
                           sub_image: bool, save_path: Path, subplot_labels: Optional[list] = None,):
    """ This function takes as input an image and a list of file (.pkl) containing masks for a given locations

    :param img:
    :param masks_path:
    :param location:
    :param square_size:
    :param n_cols:
    :param n_rows:
    :param subplot_labels:
    :param save_path:
    :return: None
    """

    if n_rows * n_cols < len(masks_path):
        raise ValueError("Grid Smaller than number of location")

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 5*n_rows))
    [ax.axis("off") for ax in axs.ravel()]
    [ax.imshow(img[location[0]-square_size:location[0]+square_size,
                   location[1]-square_size:location[1]+square_size]) for ax in axs.ravel()]

    for i, (path_, ax) in enumerate(zip(masks_path, axs.ravel())):
        if subplot_labels is not None:
            ax.set_title(subplot_labels[i])

        x_og, y_og = location
        x_range = (x_og - square_size, x_og + square_size)
        y_range = (y_og - square_size, y_og + square_size)

        with open(path_, "rb") as file:
            masks = pickle.load(file)

        for mask in masks:
            if sub_image and check_range(x_range, y_range, mask):
                x = mask[0, :] - x_range[0]
                y = mask[1, :] - y_range[0]
                ax.plot(y, x, 'r', linewidth=.8)
            elif not sub_image:
                x = mask[0, :]
                y = mask[1, :]
                ax.plot(y, x, 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(save_path / f"stardist_2d_optimization_{image_type}_{square_size}_{location}.png", dpi=300)
    plt.close()


def visualize_patches(img: np.ndarray, locations: list, square_size: int, masks: list, n_cols: int, n_rows: int,
                    save_path: Path):
    """ This function takes as input an image and corresponding masks to visualize the results at a set of given
        locations.


    :param img:
    :param locations:
    :param square_size:
    :param masks:
    :param n_cols:
    :param n_rows:
    :param save_path:
    :return:
    """

    if n_rows * n_cols < len(locations):
        raise ValueError("Grid Smaller than number of location")

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 5*n_rows))
    [ax.axis("off") for ax in axs.ravel()]

    for location, ax in zip(locations, axs.ravel()):
        for x_og, y_og in location:
            x_range = (x_og - square_size, x_og + square_size)
            y_range = (y_og - square_size, y_og + square_size)
            ax.imshow(img[y_range[0]:y_range[1], x_range[0]:x_range[1]])

            for mask in masks:
                if check_range(x_range, y_range, mask):
                    x = mask[0, :] - x_range[0]
                    y = mask[1, :] - y_range[0]
                    ax.plot(y, x, 'r', linewidth=.8)

    plt.savefig(save_path / f"stardist_2d_optimization_{image_type}_{square_size}).png")
    plt.close()


def check_range(x_range: tuple, y_range: tuple, mask: np.ndarray):
    """ Check if the masks is in the predefined range

    :param x_range: range of x coordinate
    :param y_range: range of y coordinate
    :param mask: mask given as a 2 x 'm'
    :return: True or False based on
    """
    return (((x_range[0] < mask[0, :].max() < x_range[1]) or (x_range[0] < mask[0, :].min() < x_range[1])) and
            ((y_range[0] < mask[1, :].max() < y_range[1]) or (y_range[0] < mask[1, :].min() < y_range[1])))


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
    segmentation_utils.run_segmentation_2d(path_replicate_, segmentation_type, image_type_, model_args,
                                           segment_stardist, level_, square_size, RESULTS)


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

    results_ = src_utils.get_results_path() / "segmentation" / "stardist"
    os.makedirs(results_, exist_ok=True)

    results_3d_ = results_ / "3d_seg"
    os.makedirs(results_3d_, exist_ok=True)

    return results_, results_3d_


if __name__ == "__main__":

    # TODO: Create Segmentation from The original image and perform same transformation
    #   - multiply point by the resolution
    #   - go through the wsireg transformation
    #   - check for background image

    # --------------------------------------- #
    # Script Parameters
    run = "2D"  # alternative: 3D or Patch
    square_size_ = None
    optimize = True
    image_path = "he_aligned"  # alt: dapi, he or he_aligned
    compute_masks = False
    pmin = 0.1
    pmax = 99.9
    scale = 0.2125/0.4

    # -------------------------------------- #

    # Various set up
    src_utils.check_gpu()
    results, results_3d = build_results_dir()
    init_logger()

    # Extract image path
    if image_path == "dapi":
        path_replicate = src_utils.get_mouse_xenium_path()
        model_type = "2D_versatile_fluo"
        image_dapi_type = "mip"  # alternative: focus, mip
        image = src_utils.load_image(path_replicate, img_type=image_dapi_type)
        image_type = "DAPI"

    elif image_path == "he_aligned":
        path_replicate = src_utils.get_human_breast_he_aligned_path()
        model_type = "2D_versatile_he"
        image = src_utils.load_xenium_he_ome_tiff(path_replicate, level_=0)
        image_type = "HE"

    else:
        raise ValueError("Not Implemented")

    # Run Stardist 2D optimization
    if run == "2D":

        # Image Type
        level = 0

        if optimize:
            optimize_stardist_2d(image, image_type, model_type, square_size=square_size_, compute_masks=compute_masks,
                                 save_path=results, pmin=pmin, pmax=pmax, scale=scale)

        else:
            # Model Parameters
            prob_threshold = None
            nms_threshold = None

            # Run Stardist at one location and compare to Xenium
            run_stardist_2d(path_replicate, model_type, image_type,
                            prob_thrsh=prob_threshold, nms_thrsh=nms_threshold, square_size=square_size_, )

            # Run Stardist at various locations
            run_stardist_location_2d(path_replicate, model_type, image_type,
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

        run_patch_stardist_2d(path_replicate, model_type, image_type, nms_threshold, prob_threshold)

    # Run stardist in 3D (needs some work)
    elif run == "3D":
        level = 0
        StarDist3D.from_pretrained()
        model_type = "3D_demo"
        run_stardist_3d(path_replicate, level_=level, model_type_=model_type)
