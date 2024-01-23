"""
Description: This python file has the purpose to extract use nuclei segmentation and to perform cell segmentation with
             Baysor as it was indicated to be the best follow up technique

Author: Lucas Brunschwig (lucas.brunschwig@epfl.ch)

Development:
[x] (1): Install Baysor (use installation from the GitHub) -> requires linux, hence docker environment v0.6.2
         see dockerfile_baysor for the setup
[x] (2): Code each step (load masks -> expand masks and save back -> run Baysor
[x] (3): Run script with baysor

Last Revision: 18.01
"""
import platform
from pathlib import Path
import os

import pandas as pd
import skimage
import tifffile as tiff

from src.utils import get_data_path, get_results_path
from src.nucleus_segmentation.utils import get_masks


RESULTS = Path()
if platform.system() == "Linux":
    WORKING_DIR = Path("..")
else:
    WORKING_DIR = Path("../../..")

DEFAULT_HYPERPARAMS_DATA = {
    # Taken from https://github.com/kharchenkolab/Baysor/blob/master/configs/example_config.toml

    # Name of the x column in the input data. Default: "x"
    'x': '"x_location_pixel"',
    # Name of the y column in the input data. Default: "y"
    "y": '"y_location_pixel"',
    # Name of the y column in the input data. Default: "z"
    "z": '"z_location"',
    # Ignore z-component with true
    # "force_2d": 'true',
    # Name of gene column in the input data. Default: "gene"
    "gene": '"feature_name"',
    # Minimal number of molecules per gene. Default: 1
    "min_molecules_per_gene": 1,
    # Minimal number of molecules for a cell to be considered as real.
    # It's an important parameter, as it's used to infer several other parameters. Default: 3
    "min_molecules_per_cell": 3,
}
# Additional parameters force_2d (ignore z component) / exclude_gene (n/d), confidence_nn_id (number of nearest)

DEFAULT_HYPERPARAMS_SEGMENTATION = {
    "scale": 50,
    # Standard deviation of scale across cells. Can be either number, which means absolute value of the std,
    # or string ended with "%" to set it relative to scale. Default: "25%"
    # "scale-std" : '"25%"',
    # Not exactly sure if this one should be in [Data]
    "prior_segmentation_confidence": 0.2,
    "iters": 500
}
# additional: estimate_scale_from_centers / n_clusters / nuclei_genes / cyto_genes / new_component_weights new_component_fraction

DEFAULT_HYPERPARAMS_PLOTTING = {
    # TODO: add parameters
}


def build_dir():
    results_dir = get_results_path(working_dir=WORKING_DIR)
    global RESULTS
    RESULTS = results_dir / "cell_segmentation"
    os.makedirs(RESULTS, exist_ok=True)


def run_cell_segmentation(method: str, params: dict, scale: int, expand_nuclear_area: int, transcripts_path_: Path):

    square_size_ = 8000

    # Step 1: Load nuclei segmentation based on (CellPose, Stardist, Watershed)
    print("1: Loading Masks")
    masks = get_masks(method, params={"model_": "cyto", "diameter_": 30}, img_type_="mip", square_size_=square_size_)

    # Step 2: Expand Nuclei segmentation by a given diameter and save as a tif image
    print("2: Expanding Masks")
    masks_expanded = skimage.segmentation.expand_labels(masks, distance=expand_nuclear_area)
    output_prior_segmentation = f"/tmp/prior_segmentation_{method}.tif"
    tiff.imwrite(output_prior_segmentation, masks_expanded)

    print("3: Converting Transcripts to Pixel Location")
    if not os.path.isfile(str(transcripts_path_)[:-4] + "_convert_pixels.csv")\
            or (square_size_ is not None and not os.path.isfile(str(transcripts_path_)[:-4] + "_convert_pixels_subset.csv")):
        transcripts = pd.read_csv(transcripts_path_)
        transcripts["x_location_pixel"] = (transcripts["x_location"] / 0.2125)
        transcripts["y_location_pixel"] = (transcripts["y_location"] / 0.2125)

        transcripts_path_ = Path(str(transcripts_path_)[:-4] + "_convert_pixels.csv")
        transcripts.to_csv(transcripts_path_)

        if square_size_ is not None:
            y_center = 33131 // 2  # check with max x_location (2183) > max y_location (1495)
            x_center = 48358 // 2
            # TODO: store this in metadata of masks in the future
            x_sup = x_center + square_size_ // 2
            x_inf = x_center - square_size_ // 2
            y_sup = y_center + square_size_ // 2
            y_inf = y_center - square_size_ // 2

            # Could check masks[x, y] and associate label to ensure co
            transcripts = transcripts[((transcripts['x_location_pixel'] > x_inf) & (transcripts['x_location_pixel'] < x_sup) &
                                       (transcripts['y_location_pixel'] > y_inf) & (transcripts['y_location_pixel'] < y_sup))]

            transcripts["x_location"] = transcripts["x_location"] - x_inf * 0.2125
            transcripts["y_location"] = transcripts["y_location"] - y_inf * 0.2125

            transcripts["x_location_pixel"] = transcripts["x_location_pixel"] - x_inf
            transcripts["y_location_pixel"] = transcripts["y_location_pixel"] - y_inf

            transcripts_path_ = Path(str(transcripts_path_)[:-4] + "_subset.csv")
            transcripts.to_csv(transcripts_path_)
    else:
        if square_size_ is None:
            transcripts_path_ = Path(str(transcripts_path_)[:-4] + "_convert_pixels.csv")
        else:
            transcripts_path_ = Path(str(transcripts_path_)[:-4] + "_convert_pixels_subset.csv")

    # Step 3: Use Baysor and perform cell segmentation
    print("Running Baysor")

    output = Path(RESULTS) / f"assignments_cellpose_baysor"
    os.makedirs(output, exist_ok=True)

    toml_file = output / "config.toml"

    with open(toml_file, "w") as file:
        file.write(f'[data]\n')
        for key, val in DEFAULT_HYPERPARAMS_DATA.items():
                file.write(f'{key} = {val}\n')

    with open(toml_file, "a") as file:
        file.write(f'[segmentation]\n')
        for key, val in DEFAULT_HYPERPARAMS_SEGMENTATION.items():
            file.write(f'{key} = {val}\n')

    baysor_cli = f"baysor run -c {toml_file}"
    baysor_cli += f" {str(transcripts_path_)} {str(output_prior_segmentation)}"

    print(f"Starting: Baysor with prior segmentation {method}")
    os.system(baysor_cli)
    print("Finish: Baysor")

    baysor_seg = output / "segmentation.csv"
    baysor_cell = output / "segmentation_cell_stats.csv"

    return 0


if __name__ == "__main__":
    build_dir()
    data_path = get_data_path(WORKING_DIR)
    transcripts_path = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1/transcripts.csv"

    run_cell_segmentation(method="cellpose", params={}, scale=50, expand_nuclear_area=15,
                          transcripts_path_=transcripts_path)





