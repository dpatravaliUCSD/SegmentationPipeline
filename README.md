# Cellpose 2D Distributed Nuclear Segmentation Workflow

This repository provides scripts and instructions for segmenting nuclei in large 2D microscopy images using Cellpose, with a focus on distributed segmentation for handling large datasets efficiently. Below are the steps to set up the environment, prepare training data, train a custom model, and perform segmentation.

## Setting Up the Environment

To run Cellpose and associated scripts, create a Python 3.10.16 environment with the required packages using Miniforge3 and Mamba:

1. Install Miniforge3 (if not already installed):
   - Download the installer from https://github.com/conda-forge/miniforge#download.
   - Follow the installation instructions for your OS (e.g., run the installer script on Linux/Mac or executable on Windows).

2. Create a new Mamba environment:
   mamba create -n cellpose_dev python=3.10.16

3. Activate the environment:
   mamba activate cellpose_dev

4. Install Cellpose from the GitHub repository:
   pip install git+https://www.github.com/mouseland/cellpose.git

5. Install the Cellpose GUI:
   pip install cellpose[gui]

6. Install additional dependencies:
   pip install scanpy jupyter

Your environment is now ready to run Cellpose and the scripts in this repository.

## Cloning the Repository

To get the necessary scripts, clone this repository to your local machine:

git clone https://github.com/dpatravaliUCSD/SegmentationPipeline.git
cd SegmentationPipeline


## Making Training Data

The script make_training_data.ipynb (included in this repository) generates training data by chunking a large DAPI image into smaller pieces. Follow these steps:

1. Prepare your data:
   - Change the path to the experiment directory (e.g., /mnt/sata3/Xenium_Data_Storage_2/.../output-XETG00341__0032973__mc38baseline_t1_AG001__20250109__205626/).
   - Update the experiment_path and dapi_path variables in the script to match your file locations.

2. Run the script:
   - Open make_training_data.ipynb in VScode.
   - Execute the cells sequentially. The script will:
     - Read the OME-TIFF file using pyometiff.
     - Take the maximum projection across the z-axis.
     - Split the image into 1000x1000 pixel chunks with 500-pixel overlap.
     - Save every 10th chunk as a TIFF file in the specified cellpose_training_folder (e.g., /mnt/sata4/cellpose_training/...).

3. Output:
   - Check the output folder for TIFF files (e.g., chunk_0_0.tiff, chunk_0_500.tiff, etc.) to use for training.

## Training a Custom Cellpose Model

To train a custom model for nuclear segmentation:

1. Launch the Cellpose GUI:
   Type `cellpose` in the terminal

2. Prepare training data:
   - Move the TIFF chunks from the previous step to a folder accessible by the GUI.
   - Manually annotate a subset of these images with nuclear boundaries using the GUI’s drawing tools.
   - Save each labeled mask and image as seg.npy

3. Train the model:
   - Train new model with image and masks in folder
   - Set parameters flow.threshold=0.4 for nuclei, cell probability = 0.
   - Seed with cyto2 model and use default parameters.
   - Train, and access the trained model in the new `models` folder in the directory with all of the tiff images.

## Running Nuclear Segmentation

To segment nuclei in your full dataset using the provided notebook (`nuclear_segmentation.ipynb`):

1. Prepare the segmentation notebook:
   - Load your trained Cellpose model by specifying its path (e.g., trained from the GUI step):
     from cellpose import models
     model = models.CellposeModel(model_type='/home/amonell/piloting/2025_Segmentation_Tutorial/models/CP_20250306_214544')
     Replace the path with the location of your custom model.

2. Run distributed segmentation:
   - The notebook reads the full DAPI image (morphology.ome.tif) from your experiment_path using pyometiff.OMETIFFReader.
   - It converts the image to a Zarr array with numpy_array_to_zarr for efficient distributed processing, using a chunk size of 1024x1024:
     chunk_size = (1024, 1024)
     data_zarr = numpy_array_to_zarr(os.path.join(total_path, 'DAPI_stitch.zarr'), dapi, chunks=chunk_size)
   - Distributed segmentation is performed using an altered version of Cellpose’s distributed_segmentation module (ensure CUDA_LAUNCH_BLOCKING=1 for GPU debugging if needed).
   - Execute the cells to process the entire image in blocks, stitching results across boundaries using overlap regions (default 2x nucleus diameter, approximately 60 pixels based on diameter=30).

3. Output:
   - The segmentation masks are stored in a Zarr array (e.g., labels.zarr) within the total_path directory.
   - Assign transcripts to segmented nuclei and save results:
     - Transcripts with nuclear assignments are saved as a CSV: transcripts_cellpose.csv.
     - An AnnData object with segmentation data is saved as nucleus_segmentation_adata.h5ad in the experiment_path.
   - Visualize results (optional) by plotting the full DAPI image with nuclear assignments or specific gene markers (e.g., Cd8a) using Matplotlib, as shown in the notebook.

## Why We Use Distributed Segmentation

Distributed segmentation is employed for the following reasons:

- Large Image Sizes: Microscopy images like morphology.ome.tif are often too large to process in memory on standard hardware. Distributed segmentation splits the image into manageable blocks (e.g., 1024x1024), processes them in parallel, and stitches the results.
- Efficiency: By leveraging multiple CPU/GPU cores or a cluster (e.g., via Dask), it reduces computation time significantly compared to single-threaded processing.
- Scalability: The approach scales to even larger datasets, making it suitable for high-throughput experiments.
- Stitching Challenges: As seen in the debugging process, stitching across block boundaries can fail (e.g., due to label mismatches). The distributed Cellpose workflow includes overlap regions (e.g., 2x nucleus diameter) and adjacency graph construction to merge cells split across blocks, though this requires careful tuning (e.g., avoiding aggressive label shrinking).

This repository aims to address these challenges, providing a robust pipeline for nuclear segmentation in large-scale imaging data.
