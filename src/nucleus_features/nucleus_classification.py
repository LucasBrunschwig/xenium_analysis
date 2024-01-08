# Std
from pathlib import Path
import os

# Third party
import anndata
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
import seaborn as sns

# Relative import
from utils import load_xenium_data, preprocess_transcriptomics
from gene_locus import get_gene_location
from segmentation_cellpose import load_image, segment_cellpose, transcripts_assignments
from leiden_clustering import compute_ref_labels
RESULTS = Path()


def nucleus_segmentation():
    """ Run this with ML work students"""
    raise NotImplementedError("ML Students")


def assign_transcript_to_nucleus_cellpose(adata: anndata.AnnData, save_path: str, img_path: Path,
                                          model_type: str = "cyto"):
    # check if the outputs was already generated
    if str(save_path).endswith("h5ad") and os.path.isfile(save_path):
        return anndata.read_h5ad(save_path)

    print("Segmenting...")

    # Step 1: Load Image
    img = load_image(img_path, img_type="stack", level_=0)

    # Step 2: Run Segmentation on image to extract masks
    nuclei_segmentation = segment_cellpose(img, model_type=model_type, do_3d=True)

    print("Assignments...")

    # Step 3:
    adata = transcripts_assignments(nuclei_segmentation, adata, save_path)

    return adata


def assign_transcript_to_nucleus_xenium(adata, save_path: str):
    """ This functions takes the transcripts information and assign it to a nucleus or None

    Returns: adata, the adata modified with each transcript assigned to a nucleus or None
    -------

    """

    # Check if nucleus polygons
    if not os.path.isfile(RESULTS / "nucleus_polygons.pkl"):
        cell_polygons = []
        for cell_id in adata.uns["nucleus_boundaries"]["cell_id"].unique():
            cell_vertices = adata.uns["nucleus_boundaries"][adata.uns["nucleus_boundaries"]["cell_id"] == cell_id]
            cell_vertices["point"] = cell_vertices.apply(lambda p: (int(p[1]), int(p[2])), axis=1)
            cell_polygons.append(Polygon(cell_vertices["point"].tolist()))
        with open(RESULTS / "nucleus_polygons.pkl", 'wb') as file:
            pickle.dump(cell_polygons, file)
    else:
        with open(RESULTS / "nucleus_polygons.pkl", 'rb') as file:
            cell_polygons = pickle.load(file)

    # Check if already processed
    if not os.path.isfile(save_path):
        coord = adata.uns["spots"][["x_location", "y_location", "z_location"]]
        cell_id = {}
        with tqdm(total=len(cell_polygons), desc="Processing") as pbar:
            for i, polygon in enumerate(cell_polygons):
                pbar.update(1)
                minx, miny, maxx, maxy = polygon.bounds
                coord_subset = coord[(
                        (coord["x_location"] > minx) & (coord["x_location"] < maxx) & (coord["y_location"] > miny)
                        & (coord["y_location"] < maxy))]
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
        coord["cell_id"] = coord["cell_id"].apply(lambda x: int(x[0]) if len(x) > 0 else None)
        adata.uns["spots"]["cell_id"] = coord["cell_id"]

        adata.write_h5ad(save_path)
    else:
        adata = anndata.read_h5ad(save_path)

    return adata


def label_to_color(label):
    hash_value = hash(label)
    norm_hash = (hash_value % 2 ** 32) / (2 ** 32)  # Normalize to [0, 1]
    return mcolors.hsv_to_rgb([norm_hash, 1.0, 1.0])


def visualize_plot():
    """"""
    raise NotImplementedError()


def visualize_cell_area_cluster(adata, cat: str = "nucleus_boundaries"):
    """ Do violin plots either related to cells assignments or nucleus assignments """
    # Leiden Clustering Requires to create a matrix of n_cells by n_genes and each case = number of transcripts
    # Create anndata object from this to get cell_features matrix
    # Perform preprocessing and clustering with utils function
    # Once this is we need to plot the cell location of clusters and compare it to original clustering
    # Add to the observations one column for clusters and one column for cell area -> violin plot
    # Ideally, get main marker genes for each plots

    # get_label
    adata = preprocess_transcriptomics(adata, filter_=True)
    adata.obs["leiden"] = compute_ref_labels(adata)

    cell_area = np.zeros((len(adata.uns[cat]),))

    for cell_id in adata.uns[cat]["cell_id"].unique():
        cell_vertices = adata.uns[cat][adata.uns[cat]["cell_id"] == cell_id]
        poly = Polygon(cell_vertices.copy().apply(lambda p: (int(p["vertex_x"]), int(p["vertex_y"])), axis=1).tolist())
        cell_area[int(cell_id)] = poly.area

    adata.obs["cell_area"] = cell_area[adata.obs.cell_id.to_numpy()-1]

    sns.violinplot(data=adata.obs, x="leiden", y="cell_area")
    plt.savefig(RESULTS / "violin_cluster_nucleus_area.png")


def build_nucleus_feature_matrix(transcripts_df):
    """ Requires transcriipt with each transcripts attach to a nucleus"""


def compute_cell_area(adata):
    """

    Returns
    -------

    """

    adata


def main(path_replicates_: list, panel_path_: Path, segmentation_method: str):

    # Load Gene Panel Location
    print("Computing Genome Location")
    map_loci = get_gene_location(panel_path_, organism='Mus musculus')

    # Load Annotated Data
    print("Loading Xenium Replicate")
    annotated_data = load_xenium_data(str(path_replicates_[0]) + ".h5ad", formatted=True)

    visualize_cell_area_cluster(annotated_data)

    # Select one cell and look at what it is containing
    # Work of the ML students will be extremely helpful here
    # Currently work as a cylinder
    print("Assigning Transcripts and Segmentation")
    if segmentation_method == "xenium":
        annotated_data = assign_transcript_to_nucleus_xenium(adata=annotated_data,
                                                             save_path=str(path_replicates_[0]) +
                                                                       "_assigned_transcripts.h5ad")
    elif segmentation_method == "cellpose":
        annotated_data = assign_transcript_to_nucleus_cellpose(adata=annotated_data,
                                                               save_path=str(path_replicates_[0]) +
                                                                         "_cellpose_transcripts.h5ad",
                                                               img_path=path_replicates_[0],
                                                               model_type="cyto")

    print("Plotting Results")
    # Assign transcripts type to genome location
    transcripts_assignments_ = annotated_data.uns["spots"]
    transcripts_assignments_["feature_name"] = transcripts_assignments_["feature_name"].str.upper()  # case sensitive
    map_loci.index = map_loci.index.str.upper()  # case sensitive
    # print(f"{transcripts_assignments_['cell_id'].value_counts()}")
    counts = transcripts_assignments_['cell_id'].value_counts()
    print(f"% of non-assigned transcripts {segmentation_method}-nucleus "
          f"{len(np.where(np.isnan(transcripts_assignments_['cell_id'].tolist()))[0])/len(transcripts_assignments_):.2f}")
    print(f"% of non-assigned transcripts "
          f"xenium-nucleus {1-len(np.where(transcripts_assignments_['overlaps_nucleus'].tolist())[0])/len(transcripts_assignments_):.2f}")
    print(f"% of non-assigned transcripts xenium - cells {1-annotated_data.X.toarray().sum()/len(transcripts_assignments_):.2f}")
    # # Using xenium nucleus assignment 0.65 / 0.63
    # # Using xenium cell assignment 0.22

    # Plot Transcripts per
    # plt.figure()
    # plt.grid()
    # plt.hist(counts.tolist(), color='b', bins=100)
    # plt.vlines(x=50, ymin=0, ymax=12000, color='r', label="filter nucleus < 50 transcripts")
    # plt.vlines(x=min(counts.tolist()), ymin=0, ymax=11000, color='w', label=f"min transcript = {min(counts.tolist())}")
    # plt.vlines(x=max(counts.tolist()), ymin=0, ymax=11000, color='w', label=f"max transcript = {max(counts.tolist())}")
    # plt.xlabel("Number of transcripts inside nucleus")
    # plt.ylabel("Counts")
    # plt.legend()
    # plt.title("Number transcripts per Nucleus")
    # plt.tight_layout()
    # plt.savefig(RESULTS / "Hist_Transcripts.png")
    # plt.close()

    # filter by > 50 transcripts -> 123'000 cells
    filter_index = counts[counts > 50].index.tolist()
    transcripts_assignments_ = transcripts_assignments_[transcripts_assignments_["cell_id"].isin(filter_index)]

    # we create a dictionary containing each chrom:
    chrom_closeness = {chrom1: {chrom2: 0 for chrom2 in map_loci["chrom_arm"].unique()}
                       for chrom1 in map_loci["chrom_arm"].unique()}

    chrom_distances = {chrom1: {chrom2: [] for chrom2 in map_loci["chrom_arm"].unique()}
                       for chrom1 in map_loci["chrom_arm"].unique()}

    # Visualize each cell transcripts distribution in 3D
    for cell_id in filter_index:

        # Idea to get insights about the nucleus representation:
        # - Filter cells that have less than 50 transcripts (done)
        #
        # - Try leiden clustering by transcripts inside the nucleus to see cells that look alike
        #   Compute gene groups in each cluster and look for marker genes
        #
        # - check the difference between leiden clusters when focusing on nucleus and when integrating all data
        #
        # - Nucleus um and shape for different clusters
        #
        # - Compute the average distance between chromosome associated transcripts in each cell.
        #   In a second part, compute the average of average for each pair and compare which chrom are min-max
        #   A potential plots could be a hist of these two pairs. (p-value)
        #   An additional representation is to plot the n pairs that are closest to one another and check if
        #   big chromosomes are together (done)
        #
        # - We would also like to visualize the chromosome territory it would be nice to plot the pairs that have
        #   the maximum average distance inside the nucleus. For this we can project along one angle that maximizes
        #   the distances between each chromosome or chose a plane where the distance is maximal -> registration

        # - Then, it would be interesting to observe if this organization is different depending on cell types

        cell_transcripts = transcripts_assignments_[transcripts_assignments_["cell_id"] == cell_id]
        cell_transcripts["locus"] = cell_transcripts["feature_name"].apply(
            lambda p: map_loci.loc[p].loc["chrom_arm"])

        # Compute the distance between cells
        group_locus = cell_transcripts.groupby("locus").apply(
            lambda p: np.array((p["x_location"], p["y_location"], p["z_location"])).T)
        chrom = {i: chrom_ for i, chrom_ in enumerate(group_locus.index)}
        group_locus = group_locus.tolist()
        distances = []
        for i in range(len(group_locus)):
            for j in range(i + 1, len(group_locus)):
                distance_matrix = cdist(group_locus[i], group_locus[j])
                dist = np.mean(distance_matrix)
                distances.append((chrom[i], chrom[j], dist, distance_matrix))
                chrom_distances[chrom[i]][chrom[j]].append(dist)
                chrom_distances[chrom[j]][chrom[i]].append(dist)

        # Assign the closest chromosome distance
        chrom_list = {chrom_: [None, 10000] for chrom_ in chrom_closeness.keys()}
        for group in distances:
            chrom_1, chrom_2, dist, _ = group
            if chrom_list[chrom_1][1] > dist:
                chrom_list[chrom_1] = [chrom_2, dist]
            if chrom_list[chrom_2][1] > dist:
                chrom_list[chrom_2] = [chrom_1, dist]

        for chrom_, proximity in chrom_list.items():
            if proximity[0] is not None:
                chrom_closeness[chrom_][proximity[0]] += 1

        # TO SAVE IMAGES OF CELLS (one per cluster)
        # c_map_transcripts = cell_transcripts["feature_name"].tolist()
        # c_map_transcripts_colored = [label_to_color(label) for label in c_map_transcripts]
        # c_map_chrom = cell_transcripts["locus"].tolist()
        # c_map_chrom_colored = [label_to_color(label) for label in c_map_chrom]
        #
        # x = np.array(cell_transcripts["x_location"])
        # y = np.array(cell_transcripts["y_location"])
        # z = np.array(cell_transcripts["z_location"])
        #
        # # one plot colored by transcript group
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, s=10, c=c_map_transcripts_colored, marker='o', label=c_map_transcripts)
        # ax.legend()
        # fig.savefig(RESULTS / f"id_{cell_id}_transcripts_colored.png")
        #
        # # one plot colored by chromosome group
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, s=10, c=c_map_chrom_colored, marker='o', label=c_map_chrom)
        # ax.legend()
        # fig.savefig(RESULTS / f"id_{cell_id}_locus_colored.png")

    closeness_plot = RESULTS / f"closeness_{segmentation_method}"
    os.makedirs(closeness_plot, exist_ok=True)
    for chrom_, dist_ in chrom_closeness.items():
        closest_counts = dist_.values()
        chrom_counts = dist_.keys()
        plt.figure()
        plt.bar(chrom_counts, closest_counts)
        plt.xlabel("Chromosomes")
        plt.ylabel("counts closest distance")
        plt.title(f"Closest Chromosomes Counts - chr{chrom_}")
        plt.tight_layout()
        plt.savefig(closeness_plot / f"chr{chrom_}.jpg", bbox_inches="tight")
        plt.close()

    closeness_hist_plot = RESULTS / f"closeness_hists_{segmentation_method}"
    os.makedirs(closeness_hist_plot, exist_ok=True)

    for chrom_, distances in chrom_distances.items():
        closeness_hist_plot_chrom = closeness_hist_plot / f"chr{chrom_}"
        os.makedirs(closeness_hist_plot_chrom, exist_ok=True)
        del distances[chrom_]

        test = [x for list_ in list(distances.values()) for x in list_]
        label = [y for x, list_ in distances.items() for y in [x] * len(list_)]
        violin_df = pd.DataFrame({f"chr{chrom_}": label, "dist": test})
        sns.violinplot(violin_df, x="chrom", y="dist")
        plt.title(f'Distance Violin {chrom_}')
        plt.savefig(f"violin_plot_{chrom_}.png")
        plt.close()
        # for chrom_2, values in distances.items():
        #     if len(values) > 0:
                # plt.figure()
                # plt.grid()
                # plt.hist(values, color='b', bins=50)
                # plt.vlines(x=min(values), ymin=0, ymax=1, color='w',
                #            label=f"min dist = {min(values)}")
                # plt.vlines(x=max(values), ymin=0, ymax=1, color='w',
                #            label=f"max dist = {max(values)}")
                # plt.vlines(x=np.mean(values), ymin=0, ymax=1, color='w',
                #            label=f"mean dist = {np.mean(values)}")
                # plt.xlabel("Average distance")
                # plt.ylabel("Counts")
                # plt.legend()
                # plt.title(f"Average distance between chromosome transcripts: chr{chrom_}-chr{chrom_2}")
                # plt.tight_layout()
                # plt.savefig(closeness_hist_plot_chrom / f"chr{chrom_}-chr{chrom_2}.png")
                # plt.close()

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
    mouse_brain_path = Path(r"../../scratch/lbrunsch/data/Gene_Panels"
                            r"/Xenium_V1_FF_Mouse_Brain_MultiSection_Input_gene_groups.csv")

    segment = "xenium"

    create_results_dir()

    main(path_replicates, mouse_brain_path, segment)
