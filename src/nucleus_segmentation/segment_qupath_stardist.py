"""
Description: This file will convert GeoJson masks to Polygon and optimize the parameters of the local normalization


Parameters:
    - tile size: 1024, 2048
    - normalize percentile: 0.1/99.9, 0.5, 99.5
    - resize: 0.4, 0.5, 0.6

Implementations:
[ x ]: Implement a computation of percentile normalization
[ x ]: Implement a method to compare load GeoJson object as exported from QuPath implementations -> by category
[ x ]: Implement a method to compare different features from H&E and DAPI Segmentation -> optimization of segementation
[ x ]: Implement a method to compare segmentation between H&E an DAPI -> find threshold to select high quality nuclei
[ x  ]: save nuclei from different iou bin to see how they are spatially located

"""

import os
from pathlib import Path
import json
import geojson
import scanpy as sc
from geojson import FeatureCollection, Feature, Polygon
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score

from src.utils import (load_xenium_he_ome_tiff, get_results_path, get_data_path,
                       get_human_breast_he_aligned_path, get_human_breast_dapi_aligned_path, get_geojson_masks,
                       load_geojson_background, get_human_breast_he_path, load_xenium_transcriptomics,
                       preprocess_transcriptomics)
from src.xenium_clustering.leiden_clustering import compute_ref_labels
from src.nucleus_features.nucleus_features_extraction import nucleus_transcripts_matrix_creation, convert_masks_to_df


def test_normalize_percentile_distribution(dapi_image, results_dir_):
    dapi_image = dapi_image.flatten()
    percents = [0.1, 0.5, 1.0, 5.0, 10, 95, 99.0, 99.5, 99.9, 99.99]
    percentiles = np.percentile(dapi_image, percents)

    # Plot histogram on a logarithmic scale
    plt.figure(figsize=(12, 8))
    plt.hist(dapi_image, bins=120, log=True, color='blue', alpha=0.7)
    plt.title('Histogram of DAPI Intensities on Log Scale')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency (Log Scale)')

    # Annotate percentiles on the plot
    for percentile, percent in zip(percentiles, percents):
        plt.axvline(x=percentile, color='red', linestyle='--', linewidth=1)
        plt.text(percentile, 10, f'{percentile:.1f} - {percent} - ', rotation=90, color='red')

    plt.savefig(results_dir_ / "percentile_distribution")
    plt.close()


def boxplot(keys, values, y_label, title, measure, save_file: Path = None):
    if len(keys) == 3:
        df = pd.DataFrame({
            'Category': [keys[0]] * len(values[0]) + [keys[1]] * len(values[1]) + [keys[2]] * len(values[2]),
            'Value': values[0] + values[1] + values[2]
        })
        means = [df[df["Category"] == keys[0]]['Value'].median(), df[df["Category"] == keys[1]]['Value'].median(),  df[df["Category"] == keys[2]]['Value'].median()]
        categories = [keys[0], keys[1], keys[2]]
    elif len(keys) == 2:
        df = pd.DataFrame({
            'Category': [keys[0]] * len(values[0]) + [keys[1]] * len(values[1]),
            'Value': values[0] + values[1]
        })
        means = [df[df["Category"] == keys[0]]['Value'].median(), df[df["Category"] == keys[1]]['Value'].median()]
        categories = [keys[0], keys[1]]
    elif len(keys) == 1:
        return

    # Create boxplot
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x='Category', y='Value', data=df, order=categories)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel(y_label)
    plt.ylim(-0.1*max(df.Value), 1.2*max(df.Value))

    # Perform statistical test (e.g., t-test) to determine significance
    if len(keys) == 2:
        stat, p_value = stats.mannwhitneyu(values[0], values[1], alternative="greater")
        plt.text(0.3, 0.95, f"median: {means[0]}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.6, 0.95, f"median: {means[1]}", ha='center', va='center', transform=ax.transAxes)

        if p_value < 0.05:
            plt.text(0.3, 0.9, 'p-value={:.3f} (Significant)'.format(p_value), ha='center', va='center',
                     transform=ax.transAxes)
        else:
            plt.text(0.6, 0.9, 'p-value={:.3f} (Not Significant)'.format(p_value), ha='center', va='center',
                     transform=ax.transAxes)

    elif len(keys) == 3:
        # You can perform pairwise comparisons if you have 3 groups
        # For example, compare group 1 with group 2, group 2 with group 3, and group 1 with group 3
        p_values = []
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                group1 = df[df['Category'] == categories[i]]['Value']
                group2 = df[df['Category'] == categories[j]]['Value']
                if measure == "solidity":
                    stat, p_val = stats.mannwhitneyu(group1, group2, alternative="less")
                elif measure in ["size", "circularity"]:
                    stat, p_val = stats.mannwhitneyu(group1, group2, alternative="greater")
                else:
                    stat, p_val = stats.mannwhitneyu(group1, group2)

                p_values.append(p_val)

        # Bonferroni correction for multiple comparisons
        alpha = 0.05
        adjusted_alpha = alpha / len(p_values)
        for i, p_value in enumerate(p_values):
            plt.text(0.2 + 0.3*i, 0.95, 'p-value={:.3f} (p_adj:{:.2f})'.format(p_value, adjusted_alpha), ha='center', va='center', transform=ax.transAxes)

        plt.text(0.2, 0.9, f"median: {means[0]}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.5, 0.9, f"median: {means[1]}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.8, 0.9, f"median: {means[2]}", ha='center', va='center', transform=ax.transAxes)

    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()

    plt.close()


def test_number_cells(masks_dict, params_cat, str_cat, results_dir_: Path):

    results_dir_ = results_dir_ / "count_matrix"
    os.makedirs(results_dir_, exist_ok=True)

    fixed_params = params_cat.keys()
    other_indices = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    for i, fixed in enumerate(fixed_params):
        param_option = params_cat[fixed]
        others = other_indices[i]
        for option in param_option:
            # have different categories

            cat_name_1 = list(params_cat.keys())[others[0]]
            cat_name_2 = list(params_cat.keys())[others[1]]
            params_cat_1 = params_cat[cat_name_1]
            params_cat_2 = params_cat[cat_name_2]

            count_matrix = np.zeros((len(params_cat_1), len(params_cat_2)))
            # build count matrix:
            for cat in str_cat:
                tmp = cat.split("_")
                if tmp[i] == option:
                    x = np.where(np.array(params_cat_1) == tmp[others[0]])[0][0]
                    y = np.where(np.array(params_cat_2) == tmp[others[1]])[0][0]
                    count_matrix[x, y] = len(masks_dict[cat])

            df = pd.DataFrame(count_matrix, index=params_cat_1, columns=params_cat_2)
            plt.figure(figsize=(10, 6))
            sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.1f')  # annot=True to display the count values
            plt.title(f'Number of Cells - {fixed} = {option}')
            plt.xlabel(cat_name_2)
            plt.ylabel(cat_name_1)
            plt.savefig(results_dir_ / f"{fixed}_{option}.png")


def test_feature_distribution(feature_name: str, masks_dict: dict, params_cat: dict, str_cat: list, image: np.ndarray, results_dir_: Path,):
    assert feature_name in ["size", "mean", "solidity", "circularity"]
    ylabel = {"size": "nucleus area", "mean": "mean intensity", "solidity": "solidity", "circularity": "circularity"}
    results_dir_ = results_dir_ / feature_name
    os.makedirs(results_dir_, exist_ok=True)

    fixed_params = params_cat.keys()

    for i, fixed_1 in enumerate(fixed_params):

        params_options_1 = params_cat[fixed_1]

        # Fixed a category
        for option_1 in params_options_1:
            for j, fixed_2 in enumerate(fixed_params):
                if i != j:
                    params_options_2 = params_cat[fixed_2]
                    for option_2 in params_options_2:

                        sub_cats = []
                        for cat in str_cat:
                            if str(option_1) == cat.split("_")[i] and str(option_2) == cat.split("_")[j]:
                                sub_cats.append(cat)

                        feature_by_cat = {}
                        for cat in sub_cats:
                            feature_by_cat[cat] = []
                            for mask in masks_dict[cat]:
                                if feature_name == "size":
                                    feature_by_cat[cat].append(mask.area * 0.2125**2)
                                elif feature_name == "mean":
                                    minx, miny, maxx, maxy = mask.bounds
                                    x = np.arange(int(minx), int(maxx) + 1)
                                    y = np.arange(int(miny), int(maxy) + 1)
                                    xx, yy = np.meshgrid(x, y)
                                    ix = np.vstack((yy.flatten(), xx.flatten())).T
                                    signal = image[ix[:, 0], ix[:, 1]]
                                    feature_by_cat[cat].append(np.mean(signal))
                                elif feature_name == "solidity":
                                    solidity = mask.area * 0.2125**2 / (mask.convex_hull.area * 0.2125**2)
                                    feature_by_cat[cat].append(solidity)
                                elif feature_name == "circularity":
                                    circularity =(mask.length ** 2 * 0.2125) / 4 * np.pi * mask.area * 0.2125**2
                                    feature_by_cat[cat].append(circularity)

                        values = list(feature_by_cat.values())
                        keys = list(feature_by_cat.keys())
                        save_file = results_dir_ / f"{fixed_1}{option_1}_{fixed_2}{option_2}.png"
                        boxplot(keys, values, ylabel[feature_name], f"{feature_name} Distribution", feature_name, save_file)


def test_matching(masks_he_, masks_dapi_, dapi_img_, he_img_, results_dir_):
    """

    :return:
    """

    # Create Visualization sub-directories
    results_dir_ = results_dir_
    os.makedirs(results_dir_ / "0_3-0_4", exist_ok=True)
    os.makedirs(results_dir_ / "0_4-0_5", exist_ok=True)
    os.makedirs(results_dir_ / "0_5-0_6", exist_ok=True)
    os.makedirs(results_dir_ / "0_6-0_7", exist_ok=True)
    os.makedirs(results_dir_ / "0_7-0_8", exist_ok=True)
    os.makedirs(results_dir_ / "0_8-0_9", exist_ok=True)
    os.makedirs(results_dir_ / "0_9-1_0", exist_ok=True)
    os.makedirs(results_dir_ / "no_match", exist_ok=True)

    # Keep max and min values
    min_dapi = np.min(dapi_img_)
    max_dapi = np.max(dapi_img_)

    # Convert masks to suited dataframe
    masks_he_df = pd.DataFrame(masks_he_, columns=["polygon"])
    masks_dapi_df = pd.DataFrame(masks_dapi_, columns=["polygon"])
    dapi_gpd = gpd.GeoDataFrame(masks_dapi_df, geometry=masks_dapi_df.polygon)
    dapi_sindex = dapi_gpd.sindex

    # Initialize lists
    viz_count = 0
    area_nucleus = {"no_match": [], "<0_3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    area_nucleus_dapi = {"no_match": [], "<0_3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    contained_nucleus = {"<0_3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    contained_perc = {"<0_3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    ix_iou_he = {"no_match": [], "<0_3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    ix_iou_dapi = {"<0_3": [], "0_3-0_4": [], "0_4-0_5": [], "0_5-0_6": [], "0_6-0_7": [], "0_7-0_8": [], "0_8-0_9": [], "0_9-1_0": []}
    no_match = []
    for i, mask_he in masks_he_df.iterrows():
        mask_he = mask_he.iloc[0]
        intersection = list(dapi_sindex.intersection(mask_he.bounds))

        if len(intersection) > 0:
            ious = []
            ious_index = []
            intersection_mask = dapi_gpd.iloc[intersection]
            for j, mask_dapi in intersection_mask.iterrows():
                mask_dapi = mask_dapi.polygon
                if mask_dapi.intersection(mask_he):
                    iou = mask_dapi.intersection(mask_he).area / mask_dapi.union(mask_he).area
                    ious.append(iou)
                    ious_index.append(j)
                    if viz_count < 300:
                        viz_count += 1
                        bounds = np.vstack((np.array(mask_dapi.bounds), np.array(mask_he.bounds)))
                        bounds_min = np.min(bounds[:, [0, 1]], axis=0).astype(int)
                        bounds_max = np.max(bounds[:, [2, 3]], axis=0).astype(int)

                        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

                        axs[0].imshow(dapi_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                                      origin="lower", vmin=min_dapi, vmax=max_dapi)
                        axs[1].imshow(he_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                                      extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                                      origin="lower")
                        [ax.plot(np.array(mask_he.exterior.coords)[:, 0], np.array(mask_he.exterior.coords)[:, 1],
                                 'g', zorder=2, label="he segment") for ax in axs]
                        [ax.plot(np.array(mask_dapi.exterior.coords)[:, 0], np.array(mask_dapi.exterior.coords)[:, 1],
                                 'b', zorder=2, label="dapi segment") for ax in axs]

                        if 0.3 < iou <= 0.4:
                            cat = "0_3-0_4"
                        elif 0.4 < iou <= 0.5:
                            cat = "0_4-0_5"
                        elif 0.5 < iou <= 0.6:
                            cat = "0_5-0_6"
                        elif 0.6 < iou <= 0.7:
                            cat = "0_6-0_7"
                        elif 0.7 < iou <= 0.8:
                            cat = "0_7-0_8"
                        elif 0.8 < iou <= 0.9:
                            cat = "0_8-0_9"
                        elif 0.9 < iou <= 1.0:
                            cat = "0_9-1_0"
                        else:
                            cat = None
                        if cat:
                            handles, labels = axs[0].get_legend_handles_labels()
                            fig.legend(handles, labels, loc='upper right')
                            plt.tight_layout()
                            plt.savefig(results_dir_ / cat / f"example_{i}.png")
                        plt.close()

            # If there is a confirmed intersection
            if len(ious) > 0:
                ix = np.argmax(ious)
                mask_dapi = dapi_gpd.iloc[ious_index[ix]].polygon
                contains = mask_dapi.contains(mask_he)
                iou = max(ious)
                if iou > 0.3:
                    if 0.3 < iou <= 0.4:
                        cat = "0_3-0_4"
                    elif 0.4 < iou <= 0.5:
                        cat = "0_4-0_5"
                    elif 0.5 < iou <= 0.6:
                        cat = "0_5-0_6"
                    elif 0.6 < iou <= 0.7:
                        cat = "0_6-0_7"
                    elif 0.7 < iou <= 0.8:
                        cat = "0_7-0_8"
                    elif 0.8 < iou <= 0.9:
                        cat = "0_8-0_9"
                    elif 0.9 < iou <= 1.0:
                        cat = "0_9-1_0"
                    else:
                        raise ValueError("Not Implemented")
                else:
                    cat = "<0_3"

                area_nucleus[cat].append(mask_he.area)
                area_nucleus_dapi[cat].append(mask_dapi.area)
                contained_nucleus[cat].append(contains)
                contained_perc[cat].append(mask_he.intersection(mask_dapi).area / mask_he.area)
                ix_iou_he[cat].append(i)
                ix_iou_dapi[cat].append(j)
            else:
                no_match.append(i)
                area_nucleus["no_match"].append(mask_he.area)
                ix_iou_he["no_match"].append(i)
        else:
            no_match.append(i)
            area_nucleus["no_match"].append(mask_he.area)
            ix_iou_he["no_match"].append(i)

    # ------------------------------------------------------------------------------------------------------------ #
    # Visualize no match

    viz_no_match = 0
    for i in no_match:
        if viz_no_match < 100:
            viz_no_match += 1
            mask = masks_he_df.iloc[i].polygon
            bounds = np.array(mask.bounds)
            bounds_min = bounds[[0, 1]].astype(int)
            bounds_max = bounds[[2, 3]].astype(int)

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

            axs[0].imshow(dapi_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                          extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                          origin="lower", vmin=min_dapi, vmax=max_dapi)
            axs[1].imshow(he_img_[bounds_min[1]:bounds_max[1], bounds_min[0]:bounds_max[0]],
                          extent=(bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]),
                          origin="lower")
            [ax.plot(np.array(mask.exterior.coords)[:, 0], np.array(mask.exterior.coords)[:, 1],
                     'g', zorder=2, label="he segment") for ax in axs]

            plt.savefig(results_dir_ / "no_match" / f"example_{i}.png")
            plt.close()

    total_length = [len(values) for i, values in enumerate(area_nucleus.values()) if i > 0]
    total_length = np.sum(total_length)
    print("Number of intersection: ", total_length)
    print("Percentage of intersection: ", total_length / len(masks_he_))

    # ------------------------------------------------------------------------------------------------------------ #
    # Test Size Distribution difference between match nuclei in DAPI and H&E

    plt.figure(figsize=(20, 6))
    plt.grid()
    positions = np.array(range(len(area_nucleus))) + 1
    for i, (key, values) in enumerate(area_nucleus.items()):
        if key != "no_match":
            bp = plt.boxplot(values, positions=[positions[i]], widths=0.3, patch_artist=True)
            plt.setp(bp['boxes'], color='blue')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], markeredgecolor='black')
            plt.setp(bp['medians'], color='black')

    positions = positions + 0.35
    for i, (key, values) in enumerate(area_nucleus_dapi.items()):
        if key != "no_match":
            bp = plt.boxplot(values, positions=[positions[i]], widths=0.3, patch_artist=True)
            plt.setp(bp['boxes'], color='red')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], markeredgecolor='black')
            plt.setp(bp['medians'], color='black')

    plt.ylim(-1500, 12000)

    plt.xticks(positions[1:] - 0.17, list(area_nucleus.keys())[1:])

    for i, (he, dapi) in enumerate(zip(area_nucleus.items(), area_nucleus_dapi.items())):
        he_cat, he_area = he
        dapi_cat, dapi_area = dapi

        if len(he_area) > 0 and len(dapi_area) > 0:
            u_statistic, p_value = stats.mannwhitneyu(he_area, dapi_area)
            plt.text(positions[i] - 0.25, -750, f"p_val: {p_value:.1e}", ha='center', va='center', fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    blue_patch = mpatches.Patch(color='blue', label='HE segmentation')
    red_patch = mpatches.Patch(color='red', label='DAPI segmentation')
    plt.legend(handles=[blue_patch, red_patch])
    plt.savefig(results_dir_ / "size_distribution_iou_stat_test.png")
    plt.close()

    plt.figure(figsize=(20, 6))
    plt.grid()
    positions = np.array(range(len(area_nucleus))) + 1
    for i, (key, values) in enumerate(area_nucleus.items()):
        if len(values) > 0:
            bp = plt.boxplot(values, positions=[positions[i]], widths=0.5, patch_artist=True)
            plt.setp(bp['boxes'], color='blue')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], markeredgecolor='black')
            plt.setp(bp['medians'], color='black')
    for i, he in enumerate(area_nucleus.items()):
        if i < len(area_nucleus) - 1:
            he_cat, he_area = he
            he_area_next = area_nucleus[list(area_nucleus.keys())[i+1]]
            if len(he_area) > 0:
                u_statistic, p_value = stats.mannwhitneyu(he_area, he_area_next)
                plt.text(positions[i] + 0.5, -750, f"p_val: {p_value:.1e}", ha='center', va='center', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    plt.ylim(-1200, 6000)
    plt.xticks(positions, list(area_nucleus.keys()))
    plt.title("H&E nucleus size distribution comparison")
    plt.tight_layout()
    plt.savefig(results_dir_ / "size_distribution_he_iou_stat_test.png")

    # ------------------------------------------------------------------------------------------------------------ #
    # Visualize Nucleus area distribution versus iou bins

    dfs = []
    percentage = 0
    for key, values in area_nucleus.items():
        percentage += len(values) / len(masks_he_)
        df = pd.DataFrame({"IOU bin": key + f"_({len(values)}, {percentage:.3f})", "Data": values})
        dfs.append(df)
    data_df = pd.concat(dfs)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='IOU bin', y='Data', data=data_df)
    plt.xlabel('IOU bin')
    plt.ylabel('Nucleus Area')
    plt.title('Nucleus Area Distribution by IOU bins')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir_ / "iou_bin_he.png")
    plt.close()

    # ------------------------------------------------------------------------------------------------------------ #
    # Visualize Area of H&E Contained in DAPI distribution versus iou bins

    dfs = []
    mann_whitney_u = {}
    keys = []
    percentage = 0
    for key, values in contained_perc.items():
        percentage += len(values) / (len(masks_he_)-len(area_nucleus["no_match"]))
        df = pd.DataFrame({"IOU bin": key + f"_({len(values)}, {percentage:.3f})", "Data": values})
        dfs.append(df)
        keys.append(key)

    for i, key in enumerate(contained_perc.keys()):
        if i < len(contained_perc.keys()) - 1:
            _, p_val_right = stats.mannwhitneyu(dfs[i].Data, dfs[i + 1].Data, alternative="two-sided")
            mann_whitney_u[key] = p_val_right
    data_df = pd.concat(dfs)

    plt.figure(figsize=(14, 6))
    sns.violinplot(x='IOU bin', y='Data', data=data_df)
    plt.xlabel('IOU bin')
    plt.ylabel('Nucleus Area DAPI-Contained')
    plt.title('Proportion of HE, DAPI-contained')
    plt.xticks(rotation=45)
    plt.grid()
    plt.ylim(-0.2, 1.7)

    for i, p_val in enumerate(mann_whitney_u.values()):
        plt.text(i+0.5, 1.4, f"p_val: {p_val:.2e}", ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir_ / "iou_bin_perc_he.png")
    plt.close()

    # ------------------------------------------------------------------------------------------------------------ #
    # Visualize percentage of H&E nuclei contained in DAPI nuclei distribution versus iou bins

    percentages = {}
    for category, values in contained_nucleus.items():
        true_count = sum(1 for value in values if value)
        false_count = sum(1 for value in values if not value)
        total = len(values)
        true_percentage = (true_count / total) * 100
        false_percentage = (false_count / total) * 100
        percentages[category] = {'True': true_percentage, 'False': false_percentage}

    categories = list(percentages.keys())
    true_percentages = [percentages[category]['True'] for category in categories]
    false_percentages = [percentages[category]['False'] for category in categories]

    bar_width = 0.35
    index = range(len(categories))

    plt.bar(index, true_percentages, bar_width, label='True')
    plt.bar(index, false_percentages, bar_width, bottom=true_percentages, label='False')
    plt.xlabel('Category')
    plt.ylabel('Percentage')
    plt.title('Percentage of True and False by Category')
    plt.xticks(index, categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir_ / "contains_bin_he.png")
    plt.close()

    # ------------------------------------------------------------------------------------------------------------ #
    # Save GeoJson masks for QuPath Visualization

    geojson_dir = results_dir_ / "geojson"
    os.makedirs(geojson_dir, exist_ok=True)
    for key, values in ix_iou_he.items():
        filename = key + ".geojson"
        convert_geojson(values, masks_he_df, geojson_dir / filename)


def convert_geojson(ix, masks_df, save_path_):
    masks_ = masks_df.iloc[ix]
    feature_collection = []
    for polygon in masks_.polygon:
        feature = Feature(geometry=polygon)
        feature_collection.append(feature)
    feature_collection = FeatureCollection(feature_collection)

    with open(save_path_, 'w') as outfile:
        geojson.dump(feature_collection, outfile)
    print('Finished Converting', save_path_)


def load_geojson(path: Path, shift):
    """ Expect as input a geojson object

        Format:
            type = Feature
            geometry -> type = Polygon
            properties -> object_type = detection
            properties -> classification -> name = tilesize_norm1_norm2_resolution
    """

    # Step 0: load masks
    with open(path, "r") as file:
        features_dict = json.load(file)["features"]

    # Step 1: iterate through all object and extract category:
    str_cat = []
    tile_size = []
    resolution = []
    normalization = []
    img_coordinates = []
    masks_dict = {}
    for feature in features_dict:
        if feature['type'] == 'Feature' and feature['geometry'].get('type', None) == 'Polygon':
            properties = feature['properties']
            if properties['objectType'] == 'detection':
                params_ = feature['properties']["classification"]['name']

                # tmp fix
                if len(params_.split("_")) == 4:
                    tmp = params_.split("_")
                    params_ = f"{tmp[0]}_{tmp[1]}-{tmp[2]}_{tmp[3]}"

                if params_ not in str_cat:

                    str_cat.append(params_)
                    tmp = params_.split("_")

                    if tmp[0] not in tile_size:
                        tile_size.append(tmp[0])
                    if tmp[1] not in normalization:
                        normalization.append(tmp[1])
                    if tmp[2] not in resolution:
                        resolution.append(tmp[2])

                if not masks_dict.get(params_, None):
                    masks_dict[params_] = []

                if shift: # tmp fix
                    tmp = []
                    for coord in feature['geometry']['coordinates'][0]:
                        tmp.append([coord[0]+10000, coord[1]])
                else:
                    tmp = feature['geometry']['coordinates'][0]
                masks_dict[params_].append(Polygon(tmp))

            elif properties['objectType'] == 'annotation' and len(feature['geometry']['coordinates'][0]) == 5:
                img_coordinates.append(feature['geometry']['coordinates'][0])

    params_cat = {"tile_size": tile_size, "normalization": normalization, "resolution": resolution}

    return masks_dict, str_cat, params_cat


def extract_matching_mask(masks_he_, masks_dapi_, iou_threshold_):
    """

    :param masks_he_:
    :param masks_dapi_:
    :return:
    """

    # Create DataFrame
    masks_he_df = pd.DataFrame(masks_he_, columns=["polygon"])
    masks_dapi_df = pd.DataFrame(masks_dapi_, columns=["polygon"])
    dapi_gpd = gpd.GeoDataFrame(masks_dapi_df, geometry=masks_dapi_df.polygon)
    dapi_sindex = dapi_gpd.sindex

    dict_match = {"dapi": [], "he": [], 'cell_id': []}

    for i, mask_he in masks_he_df.iterrows():
        mask_he = mask_he[0]
        intersection = list(dapi_sindex.intersection(mask_he.bounds))
        if len(intersection) > 0:
            ious = []
            ious_index = []
            intersection_mask = dapi_gpd.iloc[intersection]
            for j, mask_dapi in intersection_mask.iterrows():
                mask_dapi = mask_dapi.polygon
                if mask_dapi.intersection(mask_he):
                    iou = mask_dapi.intersection(mask_he).area / mask_dapi.union(mask_he).area
                    ious.append(iou)
                    ious_index.append(j)

            if len(ious) > 0 and max(ious) > iou_threshold:
                ix = np.argmax(ious)
                mask_dapi = dapi_gpd.iloc[ious_index[ix]].polygon
                dict_match["dapi"].append(mask_dapi)
                dict_match["he"].append(mask_he)
                dict_match["cell_id"].append(i)

    df_match = pd.DataFrame.from_dict(dict_match)

    return df_match


def visualize_clustering(adata, n_neighbors, n_pcas, results_dir):
    """ """
    adata_opt = None
    max_sil = 0
    adata_pre = preprocess_transcriptomics(adata)
    for n_neighbor in n_neighbors:
        for n_pca in n_pcas:
            adata_clustered = compute_ref_labels(adata_pre, n_neighbors=n_neighbor, n_comp=n_pca)
            _ = sc.pl.umap(adata_clustered, color='leiden', show=False)
            plt.savefig(str(results_dir / f"leiden_n_neighbors_{n_neighbor}_{n_pca}.png"))
            plt.tight_layout()
            plt.close()
            score = silhouette_score(adata_clustered.obsm['X_umap'], adata_clustered.obs['leiden'].to_numpy().astype(int))
            print(f"Silhouette score {results_dir}: {score}")
            if max_sil < score:
                max_sil = score
                adata_opt = adata_clustered.copy()

    return adata_opt


def extract_marker_genes(adata, results_dir_):
    results_dir_ = results_dir_ / "marker_genes"
    os.makedirs(results_dir_, exist_ok=True)

    adata_cp = adata.copy()
    sc.tl.rank_genes_groups(adata_cp, groupby="leiden", inplace=True)
    sc.pl.rank_genes_groups_heatmap(adata_cp, groupby="leiden", show=False, n_genes=5, show_gene_labels=True)
    plt.tight_layout()
    plt.savefig(results_dir_ / "global_heatmap.png")

    marker_genes = []
    for label in adata.obs.leiden.unique():
        sc.pl.rank_genes_groups_heatmap(adata_cp, groups=label, groupby="leiden", show=False)
        plt.tight_layout()
        plt.savefig(results_dir_ / f"heatmap_group_{label}")
        plt.close()
        marker_genes.append(sc.get.rank_genes_groups_df(adata_cp, group=label))

    return adata_cp, marker_genes


def visualize_spatial_cluster(adata, he_img, results_dir_):
    """"""
    results_dir_ = results_dir_ / "spatial_clustering"
    os.makedirs(results_dir_, exist_ok=True)

    labels = adata.obs.leiden.tolist()
    labels_unique = np.unique(labels)

    adata.obsm['X_spatial'] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    for label in labels_unique:
        _ = adata.obs[adata.obs.leiden == label].cell_id
        x_s = adata.obs[adata.obs.leiden == label].x_centroid
        y_s = adata.obs[adata.obs.leiden == label].y_centroid
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        _ = sc.pl.spatial(adata[adata.obs.leiden == label], img=he_img, scale_factor=1.0, spot_size=70, color="leiden", show=False, ax=ax[1])
        ax[0].scatter(x_s, y_s, s=1)
        ax[0].invert_yaxis()
        ax[0].axis("off")
        plt.tight_layout()
        plt.savefig(results_dir_ / f"example_{label}.png")
        plt.close()


def build_dir():
    results_path = get_results_path()
    results_path = results_path / "segment_qupath"
    os.makedirs(results_path, exist_ok=True)
    return results_path


if __name__ == "__main__":
    # ---------------------------------------------------- #
    # Run Parameters

    run_segment_main = True
    run_segment_test = False
    run_type = "MATCHING"
    iou_threshold = 0.5

    # ---------------------------------------------------- #
    # Set up the run
    results_dir = build_dir()

    masks, str_category, params_category = None, None, None
    masks_he, masks_dapi = None, None
    if run_type == "DAPI":
        # Segmentation for DAPI Images
        segmentation_path = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "morphology_mip.ome.tif - Image0_v2.geojson"
        results_dir = results_dir / "DAPI"
        os.makedirs(results_dir, exist_ok=True)
        masks, str_category, params_category = load_geojson(segmentation_path, shift=False)

    elif run_type == "H&E":
        # Segmentation for H&E Images
        segmentation_path = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "morphology_he_v2.geojson"
        results_dir = results_dir / "H&E_v2"
        os.makedirs(results_dir, exist_ok=True)
        masks, str_category, params_category = load_geojson(segmentation_path, shift=True)

    elif run_type == "MATCHING":
        segmentation_path_he = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "he_segment_full.geojson"
        segmentation_path_dapi = get_data_path() / "Xenium_Breast_Cancer_1_Qupath_Masks" / "dapi_segment_full.geojson"
        results_dir = results_dir / "matching"
        os.makedirs(results_dir, exist_ok=True)
        masks_he, str_category_he, params_category_he = load_geojson(segmentation_path_he, shift=False)
        masks_dapi, str_category_dapi, params_category_dapi = load_geojson(segmentation_path_dapi, shift=False)

    human_breast_path = get_human_breast_he_path()
    image_dapi_path = get_human_breast_dapi_aligned_path()
    image_dapi = load_xenium_he_ome_tiff(image_dapi_path, level_=0)
    image_he_path = get_human_breast_he_aligned_path()
    image_he = load_xenium_he_ome_tiff(image_he_path, level_=0)

    path_background = get_geojson_masks("background_masks")
    background = load_geojson_background(path_background)

    # ---------------------------------------------------- #
    # Main functions

    if run_segment_main:
        he_str = '4096_1.0-99.0_0.3'
        dapi_str = '4096_1.0-99.9_0.4'
        adata_filename_match = "Xenium-BreastCancer_stardist_qupath_HE_DAPI_matched.h5ad"
        adata_filename_he = f"Xenium-BreastCancer_stardist_qupath_he_{he_str}.h5ad"
        adata_filename_dapi = f"Xenium-BreastCancer_stardist_qupath_dapi_{dapi_str}.h5ad"

        if (not os.path.isfile(results_dir / "adata" / adata_filename_match) or
                not os.path.isfile(results_dir / "adata" / adata_filename_he) or
                not os.path.isfile(results_dir / "adata" / adata_filename_dapi)):
            # Running Clustering on Match between HE and DAPI optimal segmentation
            df_match = extract_matching_mask(masks_he[he_str], masks_dapi[dapi_str], iou_threshold)
            transcripts = load_xenium_transcriptomics(human_breast_path)
            df_match["polygon"] = df_match.dapi
            adata_match = nucleus_transcripts_matrix_creation(masks_nucleus=df_match[["polygon", "cell_id"]], transcripts_df=transcripts,
                                                              background_masks=background, results_dir_=results_dir,
                                                              filename_=adata_filename_match[:-5], sample_=False)
            df_match["polygon"] = df_match.he
            adata_match.uns["he_nucleus_boundaries"] = convert_masks_to_df(df_match[["polygon", "cell_id"]], adata_match.obs.cell_id.tolist())
            adata_match.write_h5ad(results_dir / "adata" / adata_filename_match)

            # Running on H&E segmentation
            df_he = pd.DataFrame(data={"polygon": masks_he[he_str], "cell_id": range(0, len(masks_he[he_str]))},
                                 columns=["polygon", "cell_id"])
            adata_he = nucleus_transcripts_matrix_creation(masks_nucleus=df_he,
                                                           transcripts_df=transcripts,
                                                           background_masks=background, results_dir_=results_dir,
                                                           filename_=adata_filename_he[:-5], sample_=False)
            # Running on DAPI segmentation
            print("Building DAPI Object")
            df_dapi = pd.DataFrame(data={"polygon": masks_dapi[dapi_str], "cell_id": range(0, len(masks_dapi[dapi_str]))},
                                   columns=["polygon", "cell_id"])
            adata_dapi = nucleus_transcripts_matrix_creation(masks_nucleus=df_dapi,
                                                             transcripts_df=transcripts,
                                                             background_masks=background, results_dir_=results_dir,
                                                             filename_=adata_filename_dapi[:-5], sample_=False)

        else:
            adata_match = sc.read_h5ad(results_dir / "adata" / adata_filename_match)
            adata_he = sc.read_h5ad(results_dir / "adata" / adata_filename_he)
            adata_dapi = sc.read_h5ad(results_dir / "adata" / adata_filename_dapi)

        # Based on Optimization
        n_neighbors = [10]
        n_pcas = [10]
        match_results = results_dir / "he_dapi_match_lab_meeting"
        os.makedirs(match_results, exist_ok=True)

        print("Running Script on DAPI-HE-Match")
        adata_match_labeled = visualize_clustering(adata_match, n_neighbors=n_neighbors, results_dir=match_results, n_pcas=n_pcas)
        extract_marker_genes(adata_match_labeled, match_results)
        visualize_spatial_cluster(adata_match_labeled, image_he, match_results)

        he_results = results_dir / "he_lab_meeting"
        os.makedirs(he_results, exist_ok=True)
        print("Running Script on HE")
        adata_he_labeled = visualize_clustering(adata_he, n_neighbors=n_neighbors, results_dir=he_results, n_pcas=n_pcas)
        extract_marker_genes(adata_he_labeled, he_results)
        visualize_spatial_cluster(adata_he_labeled, image_he, he_results)

        dapi_results = results_dir / "dapi"
        os.makedirs(dapi_results, exist_ok=True)
        print("Running Script on DAPI")
        adata_dapi_labeled = visualize_clustering(adata_dapi, n_neighbors=n_neighbors, results_dir=dapi_results, n_pcas=n_pcas)
        extract_marker_genes(adata_dapi_labeled, dapi_results)
        visualize_spatial_cluster(adata_dapi_labeled, image_he, he_results)

    elif run_segment_test:

        if run_type in ["DAPI", "H&E"]:
            # Choose the optimal segmentation
            print("Tests: segmentation QuPath")
            test_normalize_percentile_distribution(image_dapi, results_dir)
            test_number_cells(masks, params_category, str_category, results_dir)
            test_feature_distribution("size", masks, params_category, str_category, image_dapi, results_dir)
            test_feature_distribution("mean", masks, params_category, str_category, image_dapi, results_dir)
            test_feature_distribution("solidity", masks, params_category, str_category, image_dapi, results_dir)
            test_feature_distribution("circularity", masks, params_category, str_category, image_dapi, results_dir)
            # TODO: test clustering by features and compute this index that defines how well clusters are defined
            #       Based on discussion with chat-gpt -> better clustering -> link with nucleus feature extraction

        else:
            he_str = '4096_1.0-99.0_0.3'
            dapi_str = '4096_1.0-99.9_0.4'
            test_matching(masks_he[he_str], masks_dapi[dapi_str], image_dapi, image_he, results_dir)
