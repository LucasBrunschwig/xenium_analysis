import imageio
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile
import imageio

def convert_roi_to_png(input_tiff_path_, output_png_path_,
                            orig_point: tuple = (15000, 15000), square_size: int = 2000):
    try:
        # Read the OME-TIFF image
        with tifffile.TiffFile(input_tiff_path_) as tif:
            image = tif.series[0].levels[0].asarray()

        # Extract region of interest for future labelling
        print(f"Roi = square, extracted at {orig_point} from size: {square_size}x{square_size}")
        roi = image[orig_point[0]:orig_point[0] + square_size, orig_point[1]:orig_point[1] + square_size]

        # Create a figure with a specific size in inches
        fig, ax = plt.subplots(figsize=(square_size/10, square_size/10))

        # Display the region of interest
        ax.imshow(roi)
        ax.axis("off")

        # Save the image with the specified number of pixels
        plt.savefig(output_png_path_, bbox_inches='tight', pad_inches=0, dpi=100 / 7.7)

        # Safety check
        img_array = imageio.imread(output_png_path)
        print("Shape of the ROI saved:", img_array.shape)

        print(f"Conversion successful. Image saved as {output_png_path_}")
    except Exception as e:
        print(f"Error converting image: {e}")


if __name__ == "__main__":
    # Path to data
    data_path = Path("../../scratch/lbrunsch/data")
    input_tiff_path = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1/morphology_mip.ome.tif"
    output_png_path = data_path / "Xenium_V1_FF_Mouse_Brain_MultiSection_1/morphology_mip.png"

    convert_roi_to_png(input_tiff_path, output_png_path)
