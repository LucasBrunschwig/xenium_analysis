# std
import os
from pathlib import Path


# third party
import matplotlib.pyplot as plt

# relative
from src.utils import check_cuda, load_image, image_patch
from src.nucleus_segmentation.segmentation_cellpose import segment_cellpose
from src.nucleus_segmentation.segmentation_stardist import segment_stardist
from src.nucleus_segmentation.segmentaton_watershed import segment_watershed

RESULTS = Path()
RESULTS_3D = Path()


def run_predefined_models(path_replicate_: Path, run_: str, square_size_: int, image_type_: str, level_: int,
                          model_args_: dict):

    do_3d = False
    if run_ == "3D":
        image_type_ = "stack"
        do_3d = True

    # Load Image
    img = load_image(path_replicate_, img_type=image_type_, level_=level_)
    patch, boundaries = image_patch(img, square_size_=square_size_, format_="test", )

    # Step 1: simple segmentation at the center

    masks_stardist = segment_stardist(patch, do_3d=do_3d, **model_args_["stardist"])
    masks_cellpose = segment_cellpose(patch, do_3d=do_3d, **model_args_["cellpose"])
    masks_watershed = segment_watershed(patch, do_3d=do_3d, **model_args_["watershed"])

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))

    [ax.axis("off") for ax in axs]
    [ax.imshow(patch) for ax in axs]

    axs[0].set_title("Xenium Segmentation")

    axs[1].set_title(f"CellPose {model_args_['cellpose']['model_type']}")
    [axs[1].plot(mask[:, 0], mask[:, 1], 'r', linewidth=.8) for mask in masks_cellpose]

    axs[2].set_title(f"Stardist {model_args_['cellpose']['model_type']}")
    for mask in masks_stardist:
        axs[2].plot(mask[1, :], mask[0, :], 'r', linewidth=.8)

    axs[3].set_title(f"Watershed Segmentation")
    for mask in masks_watershed:
        axs[3].plot(mask[0, :], mask[1, :], 'r', linewidth=.8)

    plt.savefig(RESULTS / f"show.png")
    plt.close()


    # Step 2: Check various parameters

    # Step 3: each algorithm at various location


def build_dir():
    global RESULTS
    RESULTS = Path("../../../scratch/lbrunsch/results/nucleus_segmentation/benchmarks")
    os.makedirs(RESULTS, exist_ok=True)
    global RESULTS_3D
    RESULTS_3D = RESULTS / "3d_seg"
    os.makedirs(RESULTS_3D, exist_ok=True)


if __name__ == "__main__":

    # path replicate
    data_path = Path("../../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"

    # Set the parameters of the run
    run = "2D"
    square_size = 1000
    img_type = "mip"
    level = 0

    model_args = {
        "stardist": {"model_type_": "2D_versatile_fluo", "prob_thrsh": None, "nms_thrsh": None},
        "watershed": {},
        "cellpose": {"model_type": "cyto2", "diameter": None}
    }

    check_cuda()  # check available GPU
    build_dir()  # build results directory

    run_predefined_models(path_replicate_=path_replicate_1, run_=run, square_size_=square_size, image_type_=img_type,
                          level_=level, model_args_=model_args)
