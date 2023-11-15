import os

# Relative import
from utils import load_xenium_data

if __name__ == "__main__":

    # Path compatibility with any OS
    path = os.path.join("..", "..", "scratch", "lbrunsch", "data", "Xenium_V1_FF_Mouse_Brain_MultiSection_1")

    # Load Data
    annotated_data = load_xenium_data(path)

    # Require the cell types

    # The goal is to use DAPI information and cell types to train different models and observe the accuracy

    print("test")

