import os
import re

import pandas as pd
from Bio import Entrez
from pathlib import Path
import seaborn as sns
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Provide your email here
Entrez.email = "lucas.brunschwig@hotmail.fr"


RESULTS = Path("../../scratch/lbrunsch/results/xenium_panels")
os.makedirs(RESULTS, exist_ok=True)


def search_gene(gene_name, organism="Homo sapiens"):
    search_query = f"{gene_name}[Gene Name] AND {organism}[Organism]"
    handle = Entrez.esearch(db="gene", term=search_query, retmax=1)
    record = Entrez.read(handle)
    return record["IdList"][0]


def fetch_gene_details(gene_id):
    time.sleep(1)
    handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
    records = Entrez.read(handle)
    return records


def extract_map_locus(gene_names: list, organism: str):
    map_loci = []
    map_start = []
    map_end = []
    ids = []
    for gene_name in gene_names:
        ids.append(search_gene(gene_name, organism))
    details = fetch_gene_details(ids)
    for detail in details:
        map_locus = detail['Entrezgene_gene']['Gene-ref']['Gene-ref_maploc']
        map_loci.append(map_locus)
        # Base pair According to GRCh38.p14
        map_bp_from = detail["Entrezgene_locus"][0]["Gene-commentary_seqs"][0]['Seq-loc_int']['Seq-interval']['Seq-interval_from']
        map_bp_to = detail["Entrezgene_locus"][0]["Gene-commentary_seqs"][0]['Seq-loc_int']['Seq-interval']['Seq-interval_to']
        map_start.append(map_bp_from)
        map_end.append(map_bp_to)

    return map_loci, map_start, map_end


def extract_gene_names_from_panels(path: Path, col: str):
    panels_df = pd.read_csv(path)
    return panels_df[col].tolist()


def plot_distribution_loci(distribution, save_path: Path):
    # Assuming 'distribution' is your DataFrame
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(distribution, annot=False)  # Turn off Seaborn's annotations

    # Manually add annotations
    for i, col in enumerate(distribution.columns):
        for j, index in enumerate(distribution.index):
            ax.text(i + 0.5, j + 0.5, str(distribution.at[index, col]),
                    ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def visualize_chromosome(map_loci):
    # Colors for different chromosome stains
    color_lookup = {
        'gneg': (1., 1., 1.),
        'gpos25': (.6, .6, .6),
        'gpos50': (.4, .4, .4),
        'gpos75': (.2, .2, .2),
        'gpos100': (0., 0., 0.),
        'acen': (.8, .4, .4),
        'gvar': (.8, .8, .8),
        'stalk': (.9, .9, .9),
    }

    map_loci_formatted = {'chr' + str(i): [] for i in list(range(1, 23)) + ['X', 'Y', 'M']}
    for locus in map_loci:
        if '-' in locus:
            locus_1, locus_2 = locus.split('-')
            chrom_ = 'chr' + re.search(".*(?=p|q)", locus_1).group()
            locus_1 = re.search("(p|q).*", locus_1).group()
            map_loci_formatted[chrom_].append(locus_1)
            map_loci_formatted[chrom_].append(locus_2)
        else:
            chrom_ = 'chr' + re.search(".*(?=p|q)", locus).group()
            locus_ = re.search("(p|q).*", locus).group()
            map_loci_formatted[chrom_].append(locus_)

    ideo = pd.read_table(
        'ideogram.txt',
        skiprows=1,
        names=['chrom', 'start', 'end', 'name', 'gieStain']
    )

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(30, 12))

    max_chromosome_length = ideo.end.max()
    height_up = 0
    for i, chrom in enumerate(ideo.chrom.unique()):

        subset = ideo[ideo["chrom"] == chrom]
        # Example data
        chromosome_length = max(subset.end)
        height_down = 0.2
        height_up += 0.3

        # Draw the chromosome
        ax.broken_barh([(0, chromosome_length)], (height_up, height_down), facecolors='grey')

        for j, row in subset.iterrows():

            face_color = color_lookup[row.gieStain]
            for locus in map_loci_formatted[row.chrom]:
                if locus == row["name"]:
                    face_color = "green"
                    break

            highlight_start = row.start  # Start position of the region to highlight
            highlight_end = row.end  # End position of the region to highlight

            # Highlight the specific region
            ax.add_patch(Rectangle((highlight_start, height_up), highlight_end - highlight_start, height_down,
                                   edgecolor='black', facecolor=face_color,
                                   fill=True))
        ax.text(chromosome_length+5e6, height_up + 0.1, chrom, ha="center", va="center")  # Set the limits and labels

        if chrom == "chr22":
            break

    ax.set_xlim(0, max_chromosome_length+5e6)
    ax.set_ylim(0, height_up + 0.5)

    plt.savefig(RESULTS / "Chromosome_Human.png", bbox_inches="tight", dpi=300)


def build_df_from_locus(map_loci, map_start, map_end, gene_names, organism):

    # Split information into chromosomes and band locations
    chrom_list = []
    band_list = []
    arm_list = []
    for locus in map_loci:
        if organism == "Homo sapiens":
            if '-' in locus:
                locus_1, band_2 = locus.split('-')
                chrom_ = re.search(".*(?=p|q)", locus_1).group()
                band_1 = re.search("(p|q).*", locus_1).group()

                chrom_list.append(chrom_)
                band_list.append([band_1, band_2])
            else:
                chrom_ = re.search(".*(?=p|q)", locus).group()
                band_ = re.search("(p|q).*", locus).group()
                chrom_list.append(chrom_)
                band_list.append(band_)

            if 'p' in locus:
                arm_list.append('p')
            else:
                arm_list.append('q')
        elif organism == "Mus musculus":
            arm_list.append('q')
            chrom_list.append(locus.split(" ")[0])
            band_list = [""]

    df = pd.DataFrame(data={'chrom': chrom_list, 'arm': arm_list,
                            'band': band_list, 'seq_start': map_start,
                            'seq_end': map_end, 'raw': map_loci},
                      index=gene_names)
    df["chrom_arm"] = df.apply(lambda p: p["chrom"] + p["arm"], axis=1)

    return df


def main(file_path: Path, organism: str, plot: bool = True):

    if organism == "Homo sapiens":
        col = "Genes"
        column = ['p', 'q']
        mapping = {f"{i}": i-1 for i in range(1, 23)}
        mapping.update({"X": 22, "Y": 23})
        rows = list(mapping.keys())
    elif organism == "Mus musculus":
        col = "gene"
        column = ['q']
        mapping = {f"{i}": i - 1 for i in range(1, 20)}
        mapping.update({"X": 19, "Y": 20})
        rows = list(mapping.keys())
    else:
        raise ValueError("Undetermined organism")

    gene_names = extract_gene_names_from_panels(path=file_path, col=col)

    map_loci, map_start, map_end = extract_map_locus(gene_names, organism)

    if plot:
        distribution = np.zeros((len(rows), len(column)))

        for locus in map_loci:
            if 'p' in locus:
                ch = mapping[str(locus.split('p')[0])]
                arm = 0
            elif 'q' in locus:
                ch = mapping[str(locus.split('q')[0])]
                arm = 1
            else:
                ch = mapping[str(locus.split(' ')[0])]
                arm = 0

            distribution[ch, arm] += 1

        distribution = pd.DataFrame(data=distribution, index=rows, columns=column)

        plot_distribution_loci(distribution, save_path=RESULTS / f"{organism}.png")

        if organism == "Homo sapiens":
            visualize_chromosome(map_loci)

    df = build_df_from_locus(map_loci, map_start, map_end, gene_names, organism)

    return df


def get_gene_location(path: Path, organism: str) -> pd.DataFrame:
    """ This function returns a dataframe containing the chromosome/arm/locus location of each gene

    Parameters
    ----------
    path: the path to the set of gene panels from Xenium
    organism: 'Homo sapiens' or 'Mus musculus'

    Returns
    -------
    pandas Dataframe containing (chrom, arm, band, seq_start, seq_end, raw)

    """
    save_path = Path(str(path)[:-4] + "_map_loci.csv")
    if not os.path.isfile(save_path):
        df = main(file_path=path, organism=organism, plot=False)
        df.to_csv(save_path)
    else:
        df = pd.read_csv(save_path, index_col=0)

    return df


if __name__ == "__main__":

    # Human Gene Panels
    human_brain_path = Path(r"C:\Users\Lucas\Desktop\PhD\code\scratch\lbrunsch\data\Gene_Panels"
                            r"\Xenium_hBrain_v1_metadata.csv")
    organism = "Homo sapiens"
    main(human_brain_path, organism, plot=False)

    # Mouse Gene Panels
    mouse_brain_path = Path(r"C:\Users\Lucas\Desktop\PhD\code\scratch\lbrunsch\data\Gene_Panels"
                            r"\Xenium_V1_FF_Mouse_Brain_MultiSection_Input_gene_groups.csv")
    organism = "Mus musculus"
    main(mouse_brain_path, organism, plot=False)


