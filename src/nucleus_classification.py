# Std
from pathlib import Path
import os

# Third party
from shapely.geometry import Point, Polygon
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Relative import
from utils import load_xenium_data
from gene_locus import get_gene_location

RESULTS = Path()


def nucleus_segmentation():
    """ Run this with ML work students"""
    raise NotImplementedError()


def assign_transcript_to_nucleus(adata, tmp_path: Path = Path("../tmp")):
    """ This functions takes the transcripts information and assign it to a nucleus or None

    Returns: adata, the adata modified with each transcript assigned to a nucleus or None
    -------

    """

    os.makedirs(tmp_path, exist_ok=True)

    if not os.path.isfile(tmp_path / "nucleus_polygons.pkl"):
        cell_polygons = []
        for cell_id in adata.uns["nucleus_boundaries"]["cell_id"].unique():
            cell_vertices = adata.uns["nucleus_boundaries"][adata.uns["nucleus_boundaries"]["cell_id"] == cell_id]
            cell_vertices["point"] = cell_vertices.apply(lambda p: (int(p[1]), int(p[2])), axis=1)
            cell_polygons.append(Polygon(cell_vertices["point"].tolist()))
        with open(tmp_path / "nucleus_polygons.pkl", 'wb') as file:
            pickle.dump(cell_polygons, file)
    else:
        with open(tmp_path / "nucleus_polygons.pkl", 'rb') as file:
            cell_polygons = pickle.load(file)

    coord = adata.uns["spots"][["x_location", "y_location", "z_location"]]
    coord = coord.iloc[0:2000].copy()
    # Improve this:
    #   we know that Transcripts are ordered sequentially along x-axis
    #   we can look at polygon and check if they are ordered by location

    # one idea would be to iterate through polygons first and
    # then through transcripts by finding the x-index with a certain precision and start from there
    cell_id = {}
    with tqdm(total=len(cell_polygons), desc="Processing") as pbar:
        for i, polygon in enumerate(cell_polygons):
            pbar.update(1)
            minx, miny, maxx, maxy = polygon.bounds
            coord_subset = coord[(
                        (coord["x_location"] > minx) & (coord["x_location"] < maxx) & (coord["y_location"] > miny) & (
                            coord["y_location"] < maxy))]
            for j, row in coord_subset.iterrows():
                point = Point((row['x_location'], row['y_location']))
                if point.within(polygon):
                    if not cell_id.get(j, None):
                        cell_id[j] = []
                    cell_id[j].append(i)  # associate a transcript j to polygon i

    cell_assignment = []
    for j in coord.index:
        if cell_id.get(j, None):
            cell_assignment.append(cell_id[j])
        else:
            cell_assignment.append([])

    coord["cell_id"] = cell_assignment
    coord["cell_id"] = coord["cell_id"].apply(lambda x: x[0] if len(x) > 0 else None)  # assume one cell id per trscrpt
    adata.uns["spots"]["cell_id"] = coord["cell_id"]

    return adata


def label_to_color(label):
    hash_value = hash(label)
    norm_hash = (hash_value % 2**32) / (2**32)  # Normalize to [0, 1]
    return mcolors.hsv_to_rgb([norm_hash, 1.0, 1.0])


def visualize_plot():
    """"""
    raise NotImplementedError()


def main(path_replicates_: list, panel_path_: Path):

    # Load Gene Panel Location
    map_loci = get_gene_location(panel_path_, organism='Mus musculus')
    map_loci.rename(columns={map_loci.columns[0]: "gene"}, inplace=True)
    map_loci["chrom_arm"] = map_loci.apply(lambda p: p["chrom"] + p["arm"], axis=1)

    # Load Annotated Data
    annotated_data = load_xenium_data(path_replicates_[0], formatted=False)

    # Select one cell and look at what it is containing
    # Issue to solve how to take into account the depth when the cell boundaries is in 2-D (z-axis)
    # Work of the ML students will be extremely helpful here
    # Currently work as a cylinder
    annotated_data = assign_transcript_to_nucleus(adata=annotated_data)

    transcripts_assignments = annotated_data.uns["spots"]
    transcripts_assignments["locus"] = (transcripts_assignments["feature_name"].
                                        apply(lambda p: map_loci.loc[p]["chrom_arm"]))

    # Visualize each cell transcripts distribution in 3D
    for cell_id in transcripts_assignments["cell_id"].unique():
        cell_transcripts = transcripts_assignments[transcripts_assignments["cell_id"] == int(cell_id)]

        c_map_transcripts = cell_transcripts["feature_name"]
        c_map_transcripts_colored = [label_to_color(label) for label in c_map_transcripts]
        c_map_chrom = cell_transcripts["locus"]
        c_map_chrom_colored = [label_to_color(label) for label in c_map_chrom]

        x = np.array(cell_transcripts["x_location"])
        y = np.array(cell_transcripts["y_location"])
        z = np.array(cell_transcripts["z_location"])

        # one plot colored by transcript group
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=10, c=c_map_transcripts_colored, marker='o', label=c_map_transcripts)
        ax.legend()
        fig.savefig(RESULTS / f"id_{cell_id}_transcripts_colored")

        # one plot colored by chromosome group
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=10, c=c_map_chrom_colored, marker='o', label=c_map_chrom)
        ax.legend()
        fig.savefig(RESULTS / f"id_{cell_id}_locus_colored")

    return 0


def create_results_dir():
    global RESULTS
    RESULTS = Path("../../scratch/lbrunsch/results/nucleus_transcriptomics_3d_plots/")
    os.makedirs(RESULTS, exist_ok=True)


if __name__ == "__main__":

    # Path to data
    data_path = Path("../../scratch/lbrunsch/data")
    path_replicate_1 = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1"
    path_replicates = [path_replicate_1]

    # Human Gene Panels
    mouse_brain_path = Path(r"C:\Users\Lucas\Desktop\PhD\code\scratch\lbrunsch\data\Gene_Panels"
                            r"\Xenium_V1_FF_Mouse_Brain_MultiSection_Input_gene_groups.csv")

    create_results_dir()

    main(path_replicates, mouse_brain_path)

    print("test")
