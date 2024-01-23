# Std Library
import os
from pathlib import Path

# Third Party
import tifffile
import xml.etree.ElementTree as ET
import xmltodict
# Relative Imports
from src.utils import load_ome_tiff, get_human_breast_he_path, get_results_path, image_patch

RESULTS = Path()


def build_result_dir():
    global RESULTS
    RESULTS = get_results_path() / "he_preprocessing"

    os.makedirs(RESULTS, exist_ok=True)


def xml_to_dict(element):
    if len(element) == 0:
        return element.text
    result = {}
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in result:
            if type(result[child.tag]) is list:
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data
    return result


if __name__ == "__main__":

    build_result_dir()

    human_breast_he_path = get_human_breast_he_path()

    ome_tiff = human_breast_he_path / "additional" / "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"

    with tifffile.TiffFile(ome_tiff) as tif:
        print(f"Number of series: {len(tif.series)}")
        series = tif.series[0]
        image = series.asarray()
        metadata = xmltodict.parse(tif.ome_metadata, attr_prefix='')['OME']
        print(metadata)

    # This holds because there is only one series ! With multiple series
    dimension_order = metadata["Image"]["Pixels"]["DimensionOrder"]
    physical_size_x = float(metadata["Image"]["Pixels"]["PhysicalSizeX"])
    physical_size_y = float(metadata["Image"]["Pixels"]["PhysicalSizeY"])
    pyramidal = {}
    for key, dimensions in enumerate(metadata["StructuredAnnotations"]["MapAnnotation"]["Value"]["M"]):
        pyramidal[key] = [int(el) for el in dimensions["#text"].split(" ")]
    custom_metadata = {"dimension": dimension_order, "x_size": physical_size_x, "y_size": physical_size_y,
                       "levels": pyramidal}

