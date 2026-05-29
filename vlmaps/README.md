# VLMaps

The repository contains code for building maps with embeddings from a vision-language model (VLM), based on the [VLMaps project](https://github.com/vlmaps/vlmaps).

## Content

The following functionalities are available:

### Main functionalties

* [embeddings_from_images](embeddings_from_images) - Calculates embeddings for images in a given directory.

* [vlmaps](vlmaps) - Creates 3D vlmaps from dataset and the embeddings

* [map_to_analysis_converter](map_to_analysis_converter) - Converts various map types to analysis tool API files

* [analysis](analysis) - VLMap consistency analysis tool

### Tools, utils and scripts

* [embeddings](embeddings) - Returns CLIP embedding for a given text or image.

* [sandbox](sandbox) - Small and incomplete experiments of using VLMaps and CLIP embeddings.

* [scripts](scripts) - Short single-use programs.

* [save_camera_images](save_camera_images) - Listens to a ROS topic from a camera and saves the images in a given directory.

Instructions for using each of them are given in a README file in the corresponding folder.

## Analysis pipeline

The analysis pipeline aims to generate consistency analysis (TBD) from several datasets.

Currently supported datasets:
- Matterport3d
- *TODO*: find other dataset

Currently supported map types:
- 2D VLMap ?
- 3D VLMap

The structure of the pipeline is:
1. Generate embeddings for dataset images
2. From raw data to a map
3. From map to analysis tool API form file
4. From analysis file to analysis

Matterport 3D with 3D VLMaps:
Execute `run_all.sh $data_dir` to run the full pipeline.
This executes the following:
1. Download data.
2. Generate dataset.
3. Blur semantics.
4. Create maps.
5. Generate maps with ground-truth semantics.
6. Postprocess the created maps.
7. Run analysis.

## Setup

### Create Conda environment

Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Create a conda environment:
```
conda create -n vlmaps python=3.8
```

Activate the conda environment:
```
conda activate vlmaps
```

### Install required libraries.

Install CLIP from OpenAI:
```
pip install git+https://github.com/openai/CLIP.git
```


Install the rest of the required libraries:

```
pip install -r requirements.txt
```

## Dataset

1. Run `dataset/downloader.sh <folder>`
2. ??
3. Profit

## Map processing
**Map type:**
* REGULAR = 0
* POSTPROCESSED = 1
* INSTANCES = 2
* PREDICTED = 3
* VLMAP_INSTANCES = 4
* OUR_INSTANCES = 5
* PREDICTED_POSTPROCESSED = 6

**Processing of Ground-truth instances (GT, no need to run for each visual encoder):**
* REGULAR(0)                  --> [semantic_postprocessing]       --> POSTPROCESSED(1)
* POSTPROCESSED(1)            --> [instance_segmentation]         --> INSTANCES(2)

**Postprocessing of map with embeddings (run for vlmaps-lseg, vlmaps-openseg, ...):**
* REGULAR(0)                  --> [map_classification]            --> PREDICTED(3)
* PREDICTED(3)                --> [semantic_postprocessing]       --> PREDICTED_POSTPROCESSED(6)
* PREDICTED_POSTPROCESSED(6)  --> [instance_segmentation]         --> OUR_INSTANCES(5)
* PREDICTED(3)                --> [vlmap_instance_segmentation]   --> VLMAP_INSTANCES(4)

## Visual Encoder
Curently, two visual encoders are supported: LSeg and OpenSeg.
These are image segmentation models used for obtaining the pixelwise image embeddings from the RGB images.
They are specified in the config files: `vlmaps/config/map_config/visual_encoder/vlmaps_{encoder}.yaml`.

