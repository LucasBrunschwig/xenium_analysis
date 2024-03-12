import pickle
import json
import geojson
from geojson import FeatureCollection, Feature, Polygon
from pathlib import Path


def convert_to_geojson(masks_):

    feature_collection = []
    for mask in masks_:
        polygon = Polygon([[tuple((int(value[1]), int(value[0]))) for value in mask.T]])
        feature = Feature(geometry=polygon)
        feature_collection.append(feature)
    feature_collection = FeatureCollection(feature_collection)

    new_file = "/Users/lbrunsch/Desktop/pmin0-1_pmin99-9.geojson"
    with open(new_file, 'w') as outfile:
        geojson.dump(feature_collection, outfile)
    print('Finished', new_file)


if __name__ == "__main__":
    filename = "/Users/lbrunsch/Desktop/Phd/code/scratch/lbrunsch/results/segmentation/stardist/masks/masks_2D_versatile_he-nms0.3-prob0.3_p0.1-p99.9_scale0.425_HE-None.pkl"

    with open(filename, "rb") as file:
        masks = pickle.load(file)

    convert_to_geojson(masks)
