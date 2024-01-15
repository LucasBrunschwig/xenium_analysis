# Std
import functools
import operator
import os
import pickle
from pathlib import Path
from typing import Optional

# Third party
from cellpose import models
from cellpose.utils import outlines_list
from cellpose.contrib.distributed_segmentation import segment
from cellpose.transforms import convert_image
import matplotlib.pyplot as plt
import numpy as np
import logging
from itertools import product
import platform
import dask.array as da
import dask


class DistSegError(Exception):
    """Error in image segmentation."""

try:
    from dask_image.ndmeasure._utils import _label
    from sklearn import metrics as sk_metrics
except ModuleNotFoundError as e:
    raise DistSegError("Install 'cellpose[distributed]' for distributed segmentation dependencies") from e


if platform.system() != "Windows":
    import resource
    import sys
    # test
    # Set the maximum memory usage in bytes (300GB)
    max_memory = int(3e12)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))

    sys.path.append("..")
    sys.path.append("../..")

    from .. import utils as src_utils
    from . import utils as segmentation_utils
else:
    # Relative import
    import src.utils as src_utils
    import utils as segmentation_utils

RESULTS = Path()
RESULTS_3D = Path()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.propagate = True


def segment_cellpose(
        img_: np.ndarray,
        model_type_: str = "nuclei",
        net_avg_: bool = False,
        do_3d_: bool = False,
        diameter_: int = 30,
        chunk_: int = None,
        **kwargs
) -> np.ndarray:
    """Run cellpose and get masks

    Parameters
    ----------
    img_: Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    model_type_: model type to load
    net_avg_: evaluate 1 model or average of 4 built-in models
    do_3d_: perform 3D nuclear segmentation requires 3D array
    diameter_: estimated size of nucleus
    chunk_: requires distributed computing if images are extremely large

    Returns
    -------
    np.ndarray
        labelled image, where 0 = no masks; 1, 2, ... = mask labels
    """

    if do_3d_:
        # fast_mode, use_anisotropy, iou_depth, iou_threshold
        masks = segment(img_, channels=[0, 0], model_type=model_type_, diameter=diameter_)

    else:

        # Init model
        model = models.Cellpose(gpu=True, model_type=model_type_)

        # Eval model Parameters
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

        if chunk_ is None:
            chunk_ = img_.shape[0] // 4

        img_da = da.asarray(img_, chunks=(chunk_, chunk_))
        boundary = "none"
        image = da.overlap.overlap(img_da, depth={0: diameter_+10, 1: diameter_+10}, boundary={0: boundary, 1: boundary})
        total = None

        block_iter = zip(
            np.ndindex(*image.numblocks),
            map(
                functools.partial(operator.getitem, image),
                da.core.slices_from_chunks(image.chunks),
            ),
        )

        labeled_blocks = np.empty(image.numblocks, dtype=object)
        for index, input_block in block_iter:
            labeled_block, _, _, _ = dask.delayed(model.eval, nout=4)(x=input_block, batch_size=8, channels=[0, 0],
                                                                      net_avg=net_avg_, diameter=30, do_3D=False,
                                                                      progress=False)

            shape = input_block.shape
            labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)

            # Ensure that labels are separate
            n = labeled_block.max()
            n = dask.delayed(np.int32)(n)
            n = da.from_delayed(n, shape=(), dtype=np.int32)

            total = n if total is None else total + n

            block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
            labeled_block += block_label_offset

            labeled_blocks[index] = labeled_block
            total += n

        # Put all the blocks together
        block_labeled = da.block(labeled_blocks.tolist())

        depth = da.overlap.coerce_depth(2, {0: diameter_+10, 1: diameter_+10})

        # Actual Stitching
        iou_threshold = 0.8
        iou_depth = {0: diameter_, 1: diameter_}

        if np.prod(block_labeled.numblocks) > 1:
            # Select how much overlap do you want to join
            iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

            if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
                raise DistSegError("iou_depth (%s) > depth (%s)" % (iou_depth, depth))

            trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
            block_labeled = da.overlap.trim_internal(
                block_labeled, trim_depth, boundary=boundary
            )
            block_labeled = link_labels(
                block_labeled,
                total,
                iou_depth,
                iou_threshold=iou_threshold,
            )

            masks = da.overlap.trim_internal(
                block_labeled, iou_depth, boundary=boundary
            )

        else:
            masks = da.overlap.trim_internal(
                block_labeled, depth, boundary=boundary
            )

    return build_cellpose_mask_outlines(masks.compute())


def link_labels(block_labeled, total, depth, iou_threshold=1):
    """
    Build a label connectivity graph that groups labels across blocks,
    use this graph to find connected components, and then relabel each
    block according to those.
    """
    label_groups = label_adjacency_graph(block_labeled, total, depth, iou_threshold)
    new_labeling = _label.connected_components_delayed(label_groups)
    return _label.relabel_blocks(block_labeled, new_labeling)


def label_adjacency_graph(labels, nlabels, depth, iou_threshold):
    all_mappings = [da.empty((2, 0), dtype=np.int32, chunks=1)]
    # returns slice representing the overlap (-2*depth + 2*depth) between tile with the associate axis to look for
    slices_and_axes = get_slices_and_axes(labels.chunks, labels.shape, depth)
    for face_slice, axis in slices_and_axes:
        face = labels[face_slice]
        mapped = _across_block_iou_delayed(face, axis, iou_threshold)
        all_mappings.append(mapped)

    i, j = da.concatenate(all_mappings, axis=1)
    result = _label._to_csr_matrix(i, j, nlabels + 1)
    return result


def _across_block_iou_delayed(face, axis, iou_threshold):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_iou)
    grouped = _across_block_label_grouping_(face, axis, iou_threshold)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=np.int32)


def _across_block_label_iou(face, axis, iou_threshold):
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)

    intersection = sk_metrics.confusion_matrix(face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)

    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection

    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0)

    labels0, labels1 = np.nonzero(iou >= iou_threshold)

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])

    valid = np.all(grouped != 0, axis=0)  # Discard any mappings with bg pixels
    return grouped[:, valid]


def get_slices_and_axes(chunks, shape, depth):
    ndim = len(shape)

    # overlap depth is represented as {0: n, 1: n}
    depth = da.overlap.coerce_depth(ndim, depth)

    # get slice limit of the image
    slices = da.core.slices_from_chunks(chunks)

    # return slices and axes information
    slices_and_axes = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - 2 * depth[ax], sl[ax].stop + 2 * depth[ax]
            )
            slices_and_axes.append((tuple(slice_to_append), ax))
    return slices_and_axes


def build_cellpose_mask_outlines(masks):

    masks_outline = outlines_list(masks, multiprocessing=False)
    for i, mask in enumerate(masks_outline):
        masks_outline[i] = mask.T
    return masks_outline


def optimize_cellpose_2d(path_replicate_: Path, img_type_: str, square_size_: Optional[int],
                         compute_masks: bool = True):

    # Loading Image
    print(f"Loading Images: {img_type_} with size {square_size_}")
    img = src_utils.load_image(path_replicate_, img_type=img_type_, level_=0)
    patch, boundaries = src_utils.image_patch(img, square_size_=square_size_)

    # Potential Parameters
    model_version_ = ["cyto", "cyto2", "nuclei"]
    diameters = [7, 15, 30]
    comb = product(model_version_, diameters)

    masks_dir = RESULTS / "masks"
    os.makedirs(masks_dir, exist_ok=True)

    if compute_masks:
        print("Start Segmenting")
        for model_, diameter_ in comb:
            print(f"Segment: model-{model_} and diameter-{diameter_}")
            masks_cellpose = segment_cellpose(patch.copy(), model_type_=model_, do_3d_=False, diameter_=diameter_,
                                              distributed_=False)

            with open(masks_dir / f"masks_{model_}-diameter{diameter_}"
                                  f"_{img_type_}-{square_size}.pkl", 'wb') as file:
                pickle.dump(masks_cellpose, file)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
    [ax.axis("off") for ax in axs.ravel()]

    if square_size is not None:
        [ax.imshow(patch) for ax in axs.ravel()]
    else:  # small region to plot
        og = (patch.shape[0]//2 - 400, patch.shape[0]//2 + 400)
        [ax.imshow(patch[og[0]:og[1], og[0]:og[1]]) for ax in axs.ravel()]

    comb = product(model_version_, diameters)
    for ax, (model_, diameter_) in zip(axs.ravel(), comb):
        ax.set_title(f"Model: {model_}, Diam: {diameter_}")

        with open(masks_dir / f"masks_{model_}-diameter{diameter_}"
                              f"_{img_type_}-{square_size}.pkl", 'rb') as file:
            masks_cellpose = pickle.load(file)

        for mask in masks_cellpose:
            if square_size is not None or (square_size is None and
                                           (((og[0] < mask[0, :].max() < og[1]) or
                                            (og[0] < mask[0, :].min() < og[1])) and
                                           ((og[0] < mask[1, :].max() < og[1]) or
                                            (og[0] < mask[1, :].min() < og[1])))):
                ax.plot(mask[0, :], mask[1, :], 'r', linewidth=.8)

    plt.tight_layout()
    plt.savefig(RESULTS / f"cellpose_2d_optimization_{img_type_}_{square_size}.png")
    plt.close()

#
#
# def run_cellpose_2d(path_replicate: Path, img_type: str = "mip"):
#     """ This function run cellpose on an image
#
#     Parameters
#     ----------
#     path_replicate (Path): path to the replicate
#     img_type (str): type of images
#
#     Returns
#     -------
#
#     """
#
#     # Load Image Type
#     img = src_utils.load_image(path_replicate, img_type)
#
#     # Returns a test patch with the image boundaries
#     patch, boundaries = src_utils.image_patch(img, square_size_=700, format_="test")
#
#     adata = src_utils.load_xenium_data(Path(str(path_replicate) + ".h5ad"))
#
#     # Convert xenium predefined nucleus boundaries to pixels locations
#     # (x,y): vertex_x is the horizontal axis / vertex y is the vertical axis
#     # from_metadata 1 pixel = 0.2125 microns
#     x_conversion = 0.2125
#     y_conversion = 0.2125
#     adata.uns["nucleus_boundaries"]["vertex_y_pixel"] = adata.uns["nucleus_boundaries"]["vertex_y"].apply(
#         lambda p: round(p/y_conversion))
#     adata.uns["nucleus_boundaries"]["vertex_x_pixel"] = adata.uns["nucleus_boundaries"]["vertex_x"].apply(
#         lambda p: round(p/x_conversion))
#
#     # Selection of segmented nucleus that are inside the patch
#     pix_boundaries = adata.uns["nucleus_boundaries"][(adata.uns["nucleus_boundaries"]["vertex_x_pixel"] > boundaries[1][0]) &
#                                                      (adata.uns["nucleus_boundaries"]["vertex_x_pixel"] < boundaries[1][1]) &
#                                                      (adata.uns["nucleus_boundaries"]["vertex_y_pixel"] > boundaries[0][0]) &
#                                                      (adata.uns["nucleus_boundaries"]["vertex_y_pixel"] < boundaries[0][1])
#                                                      ]
#
#     # Run CellPose with 3 predefined models (nuclei, cyto, cyto2)
#     seg_patch_nuclei = segment_cellpose(patch, model_type_="nuclei", distributed_=distributed_)
#     seg_patch_nuclei_outlines = outlines_list(seg_patch_nuclei, multiprocessing=False)
#
#     seg_patch_cyto = segment_cellpose(patch, model_type_="cyto", distributed_=distributed_)
#     seg_patch_cyto_outlines = outlines_list(seg_patch_cyto, multiprocessing=False)
#
#     seg_patch_cyto2 = segment_cellpose(patch, model_type_="cyto2", distributed_=distributed_)
#     seg_patch_cyto2_outlines = outlines_list(seg_patch_cyto2, multiprocessing=False)
#
#     seg_patch_comb = segment_cellpose(patch, net_avg_=True, distributed_=distributed_)
#     seg_patch_comb_outlines = outlines_list(seg_patch_comb, multiprocessing=False)
#
#     plt.imshow(patch)
#     plt.savefig(RESULTS / f"og_patch_{img_type}.png", dpi=500)
#     plt.close()
#
#     # Plot the results and compare it to the original images
#     fig, ax = plt.subplots(3, 3, figsize=(20, 15))
#     plt.subplots_adjust(hspace=0.3)
#     fig.suptitle(f"Nucleus Segmentation for Pretrained Models on {img_type.upper()} morphology")
#     [x_.axis("off") for x_ in ax.ravel()]
#     [x_.imshow(patch) for x_ in ax.ravel()]
#     ax[0, 1].set_title("Original DAPI Image")
#     [ax[1, 0].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5) for mask in seg_patch_nuclei_outlines]
#     ax[1, 0].set_title("CellPose - Nucleus")
#     [ax[1, 1].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5) for mask in seg_patch_cyto_outlines]
#     ax[1, 1].set_title("CellPose - Cyto")
#     [ax[1, 2].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5) for mask in seg_patch_cyto2_outlines]
#     ax[1, 2].set_title("CellPose - Cyto2")
#
#     # Plot Xenium original boundaries
#     [ax[2, i].set_title("Xeinum Segmentation") for i in [0, 1, 2]]
#     for cell_seg in pix_boundaries["cell_id"].unique():
#         x = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy() - boundaries[1][0]
#         y = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy() - boundaries[0][0]
#         ax[2, 0].plot(x, y, c='r', linewidth=.5)
#         ax[2, 1].plot(x, y, c='r', linewidth=.5)
#         ax[2, 2].plot(x, y, c='r', linewidth=.5)
#
#     plt.tight_layout()
#     fig.savefig(RESULTS / f"cellpose_{img_type}_segmentation.png", bbox_inches="tight", dpi=500)
#
#     fig, ax = plt.subplots(1, 3)
#     [x_.axis("off") for x_ in ax]
#     [x_.imshow(patch) for x_ in ax]
#     ax[0].set_title("CellPose - Average")
#     ax[1].set_title("Xenium Segmentation")
#     ax[2].set_title("Segmentation Superposition")
#     for cell_seg in pix_boundaries["cell_id"].unique():
#         x = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_x_pixel"].to_numpy() - boundaries[1][0]
#         y = pix_boundaries[pix_boundaries["cell_id"] == cell_seg]["vertex_y_pixel"].to_numpy() - boundaries[0][0]
#         ax[1].plot(x, y, c='r', linewidth=.5)
#         ax[2].plot(x, y, c='r', linewidth=.5)
#     [ax[0].plot(mask[:, 0], mask[:, 1], 'aqua', linewidth=.5, alpha=0.5) for mask in seg_patch_comb_outlines]
#     [ax[2].plot(mask[:, 0], mask[:, 1], 'aqua', linewidth=.5, alpha=0.5) for mask in seg_patch_comb_outlines]
#     plt.tight_layout()
#     fig.savefig(RESULTS / f"superposition_xenium_cellpose_{img_type}.png", dpi=500)
#
#     return 0
#
#
# def run_cellpose_3d(path_replicate_: Path, level_: int = 0, diameter_: int = 10):
#
#     img = src_utils.load_image(path_replicate_, img_type="stack", level_=level_)
#     patch, boundaries = src_utils.image_patch(img, square_size_=700)
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
#     if not os.path.isfile(RESULTS_3D / "mask_outline.pkl"):
#         seg_3d = segment_cellpose(img, do_3d_=True, diameter_=diameter_, distributed_=distributed_)
#         with open(RESULTS_3D / f"mask_level{level_}_diameter{diameter_}.pkl", "wb") as file:
#             pickle.dump(seg_3d, file)
#         seg_3d_outlines = outlines_list(seg_3d, multiprocessing=False)
#         with open(RESULTS_3D / f"mask_outline{level_}_diameter{diameter_}.pkl", "wb") as file:
#             pickle.dump(seg_3d_outlines, file)
#     else:
#         with open(RESULTS_3D / f"mask_outline{level_}_diameter{diameter_}.pkl", "rb") as file:
#             seg_3d_outlines = pickle.load(file)
#
#     print("Plotting Resulting Segmentation")
#
#     seg_3d_outlines = seg_3d_outlines[boundaries[1][0]:boundaries[1][1],
#                                       boundaries[2][0]:boundaries[2][1]]
#
#     fig, axs = plt.subplots(3, 4)
#
#     for i, (layer, ax) in enumerate(zip(patch, axs.ravel())):
#         ax.axis("off")
#         ax.set_title(f"nucleus segmentation - layer {i}")
#         ax.imshow(layer)
#         [ax[i].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.5, alpha=1) for mask in seg_3d_outlines[i, :, :]]
#
#     plt.tight_layout()
#     fig.savefig(RESULTS_3D / f"3d_patch_segmentation_level{level_}_diameter{diameter_}.png", dpi=600)
#
#     return 0
#
#
# def transcripts_assignments(masks: np.ndarray, adata, save_path: str, qv_cutoff: float = 20.0):
#     """ This is a function that will be moved in another section but used testing nucleus transcripts assignments """
#
#     transcripts_df = adata.uns["spots"]
#
#     mask_dims = {"z_size": masks.shape[0], "x_size": masks.shape[2], "y_size": masks.shape[1]}
#
#     # Iterate through all transcripts
#     transcripts_nucleus_index = []
#     for index, row in transcripts_df.iterrows():
#
#         x = row['x_location']
#         y = row['y_location']
#         z = row['z_location']
#         qv = row['qv']
#
#         # Ignore transcript below user-specified cutoff
#         if qv < qv_cutoff:
#             continue
#
#         # Convert transcript locations from physical space to image space
#         pix_size = 0.2125
#         z_slice_micron = 3
#         x_pixel = x / pix_size
#         y_pixel = y / pix_size
#         z_slice = z / z_slice_micron
#
#         # Add guard rails to make sure lookup falls within image boundaries.
#         x_pixel = min(max(0, x_pixel), mask_dims["x_size"] - 1)
#         y_pixel = min(max(0, y_pixel), mask_dims["y_size"] - 1)
#         z_slice = min(max(0, z_slice), mask_dims["z_size"] - 1)
#
#         # Look up cell_id assigned by Cellpose. Array is in ZYX order.
#         nucleus_id = masks[round(z_slice), round(y_pixel), round(x_pixel)]
#
#         # If cell_id is not 0 at this point, it means the transcript is associated with a cell
#         if nucleus_id != 0:
#             # Increment count in feature-cell matrix
#             transcripts_nucleus_index.append(nucleus_id)
#         else:
#             transcripts_nucleus_index.append(None)
#
#     adata.uns["spot"]["nucleus_id"] = transcripts_nucleus_index
#     if str(save_path).endswith("h5ad"):
#         adata.write_h5ad(save_path)
#     else:
#         adata.write_h5ad(str(save_path)+".h5ad")
#
#     return adata


def build_results_dir():
    global RESULTS

    if platform.system() != "Windows":
        working_dir = Path("..")
    else:
        working_dir = Path("../../..")
    RESULTS = working_dir / "scratch/lbrunsch/results/nucleus_segmentation/cellpose"
    os.makedirs(RESULTS, exist_ok=True)
    global RESULTS_3D
    RESULTS_3D = RESULTS / "3d_segmentation"
    os.makedirs(RESULTS_3D, exist_ok=True)


if __name__ == "__main__":

    # Various set up
    src_utils.check_cuda()
    build_results_dir()
    init_logger()

    # Run Parameters
    run = "2D"  # alternative: 3D or Patch
    square_size = None
    optimize = True

    # Path
    if platform.system() != "Windows":
        working_dir = Path("..")
    else:
        working_dir = Path("../../..")
    data_path = working_dir / "scratch/lbrunsch/data"
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    if run == "2D":

        # Model Version
        model_type = ""

        # Image Type
        image_type = "mip"  # alternative: focus, mip
        level = 0

        if optimize:
            print("Running 2D Optimization on Nuclei")
            optimize_cellpose_2d(path_replicate_1, image_type, square_size_=square_size)

    #     else:
    #
    #         # Model Parameters
    #         model_version = "cyto2"
    #         diameter = 12
    #
    #         # Run CellPose at one location and compare to Xenium
    #         run_cellpose_2d(path_replicate_1, image_type, square_size=square_size_, diameter=diameter)
    #
    #         # Run CellPose at various locations
    #         run_cellpose_location_2d(path_replicate_1, model_type, image_type,
    #                                  prob_thrsh=prob_threshold, nms_thrsh=nms_threshold, square_size=square_size_)
    #
    # # Run stardist with various patch size
    # elif run == "patch":
    #     StarDist2D.from_pretrained()
    #
    #     # Image Type
    #     image_type = "mip"  # alternative: focus, mip
    #
    #     # Model Parameters
    #     model_type = "2D_versatile_fluo"  # alternative: 2D_versatile_fluo, 2D_paper_dsb2018, 2D_versatile_he, 2D_demo
    #     prob_threshold = None
    #     nms_threshold = None
    #
    #     run_patch_cellpose_2d(path_replicate_1, model_type, image_type, nms_threshold, prob_threshold)
    #
    # # Run Stardist in 3D (needs some work)
    # elif run == "3D":
    #     level = 0
    #     run_cellpose_3d(path_replicate_1, level_=level)

