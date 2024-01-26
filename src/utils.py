# std
import os
from pathlib import Path
import shutil
import zipfile
import re

# third party
import anndata
import scanpy as sc
import pandas as pd
import gzip
import squidpy as sq
import numpy as np
import tifffile
import torch
import json
import xmltodict


def load_config():

    # Get Absolute Path
    path = Path(f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1]))

    with open(path / "config.json", 'r') as file:
        config = json.load(file)

    config["absolute_path"] = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-3])

    return config


def decompress(path_file, extension='gz'):
    if not os.path.isfile(f'{path_file}'):
        if extension == 'gz':
            with gzip.open(f'{path_file}.gz', 'rb') as f_in:
                with open(f'{path_file}', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif extension == 'zip':
            with zipfile.ZipFile(f'{path_file}.zip', 'r') as f_in:
                with open(f'{path_file}', 'w') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def format_xenium_adata(path: Path, output_path: Path,
                      segmentation_info: bool = True,
                      cluster_info: bool = True):
    """

    Parameters
    ----------
    path (Path): Path to the folder containing Xenium output
    output_path (str): output_path where to save the formatted adata file
    segmentation_info (bool): loads the nuclear and cellular boundaries
    cluster_info (bool): loads the predefined clustering (graph, kmeans (1-10))

    Returns
    -------
    adata object

    """
    if not os.path.isfile(Path(str(output_path) + '.h5ad')):
        # decompress various files
        decompress(path / 'transcripts.csv')
        decompress(path / 'cell_boundaries.csv')
        decompress(path / 'nucleus_boundaries.csv')
        decompress(path / 'cells.csv')

        # read data that contains cell feature matrix and some additional information
        adata = sc.read_10x_h5(path / "cell_feature_matrix.h5")

        cell = pd.read_csv(path / "cells.csv")

        # Integrate cells.csv in adata object
        cell.set_index(adata.obs_names, inplace=True)
        adata.obs = cell
        adata.obs["og_index"] = adata.obs.index
        adata.obs_names_make_unique()

        # Format Spatial information for plotting
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

        # Ensure unique index for gene by focusing on gene_ids
        adata.var["SYMBOL"] = adata.var.index
        adata.var.set_index("gene_ids", drop=True, inplace=True)

        # Specific information about transcript
        transcripts = pd.read_csv(path / 'transcripts.csv', index_col=0)
        transcripts = transcripts[~transcripts["feature_name"].str.contains("blank", case=False)]
        transcripts = transcripts[~transcripts["feature_name"].str.contains("negcontrol", case=False)]

        adata.uns['spots'] = transcripts

        # Retrieve UMAP, TSNE, PCA information
        additional_cat = ["X_umap", "X_tsne", "X_pca"]
        additional_paths = ['/analysis/umap/gene_expression_2_components/projection.csv',
                            '/analysis/tsne/gene_expression_2_components/projection.csv',
                            '/analysis/pca/gene_expression_10_components/projection.csv']
        for cat, sub_path in zip(additional_cat, additional_paths):
            try:
                X = pd.read_csv(Path(str(path) + sub_path))
                adata.obsm[cat] = np.array(X)
            except Exception as e:
                print(f"Failed retrieving: {additional_cat}")

        if segmentation_info:
            nuclear_boundaries = pd.read_csv(path / "nucleus_boundaries.csv")
            adata.uns['nucleus_boundaries'] = nuclear_boundaries
            cell_boundaries = pd.read_csv(path / "cell_boundaries.csv")
            adata.uns['cell_boundaries'] = cell_boundaries

        if cluster_info:
            # Retrieve Clustering and differential expression values
            additional_cat = (['graph_clusters'] +
                              [f'kmeans{k}_clusters' for k in range(2, 11)])
            additional_paths = (['/gene_expression_graphclust/'] +
                                [f'/gene_expression_kmeans_{k}_clusters/' for k in range(2, 11)])
            for main_folder, filename in zip(['clustering', 'diffexp'], ['clusters.csv', 'differential_expression.csv']):
                for cat, sub_path in zip(additional_cat, additional_paths):
                    try:
                        path_ = Path(str(path) + f'/analysis/' + main_folder + sub_path + filename)
                        X = pd.read_csv(path_, index_col=0)
                        if main_folder == "clustering":
                            adata.obs[cat] = list(X["Cluster"].astype(str))
                        elif main_folder == "diffexp":
                            if cat == 'graph_clusters':
                                k_mean_n = adata.obs[cat].to_numpy().astype(int).max()
                            else:
                                k_mean_n = int(re.search("\d+", sub_path).group())
                            for j in range(1, k_mean_n + 1):
                                adata.var[cat + f"_cluster{j}_log2_fold"] = X[f"Cluster {j} Log2 fold change"].tolist()
                                adata.var[cat + f"_cluster{j}_ajusted_pvalue"] = X[f"Cluster {j} Adjusted p value"].tolist()
                    except Exception as e:
                        print(f"Error: {e}")

        # Save h5ad file to specified location
        adata.write(Path(str(output_path) + '.h5ad'))
    else:
        adata = anndata.read_h5ad(Path(str(output_path) + '.h5ad'))

    return adata


def load_xenium_images(path: Path, type: str = None):
    """

    Parameters
    ----------
    path
    type

    Returns
    -------

    """
    results = []
    if type is None or type == "morphology":
        results.append(tifffile.imread(str(path / 'morphology.ome.tif')))
    if type is None or type == "focus":
        results.append(tifffile.imread(str(path / 'morphology_focus.ome.tif')))
    if type is None or type == "mip":
        results.append(tifffile.imread(str(path / 'morphology_mip.ome.tif')))
    return results


def load_xenium_data(path: Path):
    """

    Parameters
    ----------
    path: if formatted expect a .h5ad, else the main sample folder
    formatted: if loading the formatted dataset

    Returns
    -------

    """

    if str(path).endswith(".h5ad"):
        path_file = path
        path_data = Path(str(path)[:-5])
    else:
        path_data = path
        path_file = Path(str(path)+".h5ad")

    if os.path.isfile(path_file):
        adata = anndata.read_h5ad(path_file)
    elif os.path.isdir(path_data):
        adata = format_xenium_adata(path_data, output_path=path_data)
    else:
        raise ValueError(f"Path does not exist {path}")

    return adata


def load_rna_seq_data(path):
    adata = sc.read_loom(path)

    # Ensure unique var names
    adata.var["SYMBOL"] = adata.var.index
    adata.var.set_index("Accession", drop=True, inplace=True)

    adata.obs["og_index"] = adata.obs.index
    adata.obs_names_make_unique()

    # Mark the 'mt' gene
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]

    return adata


def preprocess_transcriptomics(adata, filter_: bool = True):
    """Perform normalization on transcriptomics data obtained through xenium

        (Opt) Filtering min counts, min cells (1) Normalize total (2) log(X+1)

        Remark: in some cases you do not want the preprocessing to drop data points (set filter_ = False)
    """

    # Filter adata by number of counts per cell and number of gene abundance across cells
    if filter_:
        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_genes(adata, min_cells=5)

    # Normalize
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    return adata


def plot_xenium_labels(adata, label_key):
    # Spatial Distribution of counts
    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        shape=None,
        color=[
            label_key,
        ],
        wspace=0.4,
    )


def get_name_from_path(path: Path) -> str:
    return str(path).split(os.sep)[-1]


def load_image(path_replicate: Path, img_type: str, level_: int = 0):
    if img_type == "mip":
        img_file = str(path_replicate / "morphology_mip.ome.tif")
        with tifffile.TiffFile(img_file) as tif:
             image = tif.series[0].levels[level_].asarray()
    elif img_type == "focus":
        img_file = str(path_replicate / "morphology_focus.ome.tif")
        with tifffile.TiffFile(img_file) as tif:
            image = tif.series[0].levels[level_].asarray()
    elif img_type == "stack":
        img_file = str(path_replicate / "morphology.ome.tif")
        with tifffile.TiffFile(img_file) as tif:
             image = tif.series[0].levels[level_].asarray()
    else:
        raise ValueError("Not a type of image")

    print("Image shape:", image.shape)
    return image


def image_patch(img_array_, type_: str, square_size_: int = 400, orig_: tuple = None):
    """

    Parameters
    ----------
    img_array_: 2- or 3-dimensional array
    square_size_: the length of the image square
    format_: "test" returns one square patch at the image center (width = square size)
             "training": returns a list of patches adapting the square size to match the image size
    orig_: choose the origin of the square expected [x, y] or [z, y, x]

    Returns
    -------
    returns: list of patches or one patch as np.ndarray

    """

    # if patch is false return the original image with its shape
    if square_size_ is None:
        if len(img_array_.shape) == 2:
            return [img_array_, [[0, img_array_.shape[0]], [0, img_array_.shape[1]]]]
        else:
            return [img_array_, [[0, img_array_.shape[0]], [0, img_array_.shape[1]], [0, img_array_.shape[2]]]]

    if type_ == "HE":
        coord_1, coord_2 = 0, 1
    elif type_ == "DAPI":
        if len(img_array_.shape) == 2:
            coord_1, coord_2 = 0, 1
        else:
            coord_1, coord_2 = 1, 2
    else:
        raise ValueError("Unknown Image Type")

    # if no coordinates take the center
    if orig_ is None:
        l_t = img_array_.shape[coord_1] // 2 - square_size_ // 2
        r_t = img_array_.shape[coord_1] // 2 + square_size_ // 2
        l_b = img_array_.shape[coord_2] // 2 - square_size_ // 2
        r_b = img_array_.shape[coord_2] // 2 + square_size_ // 2

    # use specified coordinates
    else:
        l_t = orig_[coord_1] - square_size_ // 2
        r_t = orig_[coord_1] + square_size_ // 2
        l_b = orig_[coord_2] - square_size_ // 2
        r_b = orig_[coord_2] + square_size_ // 2

    if len(img_array_.shape) == 2 or type_ == "HE":
        return [img_array_[l_t:r_t, l_b:r_b], ([l_t, r_t], [l_b, r_b])]
    else:
        return [img_array_[:, l_t:r_t, l_b:r_b],
                ([0, l_t, r_t], [img_array_.shape[0], l_b, r_b])]



def check_gpu():
    device = None
    if torch.cuda.is_available():
        print("GPU available", torch.cuda.current_device())
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS Device Found")
    else:
        print("No GPU available")
    return device


def get_data_path(working_dir: Path = None) -> Path:
    if working_dir is None:
        config = load_config()
        working_dir = Path(config["absolute_path"])
    return working_dir / "scratch/lbrunsch/data"


def get_results_path(working_dir: Path = None) -> Path:
    if working_dir is None:
        config = load_config()
        working_dir = Path(config["absolute_path"])
    return working_dir / "scratch/lbrunsch/results"


def get_working_dir():
    return load_config()["absolute_path"]


def get_mouse_xenium_path():
    config = load_config()
    return get_data_path() / config["mouse_replicate_1"]


def get_human_breast_he_path():
    config = load_config()
    return get_data_path() / config["human_breast_replicate_1"]


def load_xenium_he_ome_tiff(path: Path, level_: int, debug: bool = False):

    if debug:
        print(f"Loading: {path.name}, level={level_}")
    with tifffile.TiffFile(path) as tif:
        if debug:
            print(f"\tNumber of series: {len(tif.series)}")
        image = tif.series[0].levels[level_].asarray()
        metadata = xmltodict.parse(tif.ome_metadata, attr_prefix='')['OME']

    # This holds because there is only one series ! With multiple series
    dimension_order = metadata["Image"]["Pixels"]["DimensionOrder"]
    physical_size_x = float(metadata["Image"]["Pixels"]["PhysicalSizeX"])
    physical_size_y = float(metadata["Image"]["Pixels"]["PhysicalSizeY"])
    pyramidal = {}
    for key, dimensions in enumerate(metadata["StructuredAnnotations"]["MapAnnotation"]["Value"]["M"]):
        pyramidal[key] = [int(el) for el in dimensions["#text"].split(" ")]
    custom_metadata = {"dimension": dimension_order, "x_size": physical_size_x, "y_size": physical_size_y,
                       "levels": pyramidal}

    print("Image shape:", image.shape)

    return image, custom_metadata
