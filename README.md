This project is a combined neural network utilizing an spiking CNN with backpropagation and YOLOv3 for object detection. 

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/Zaabon/spiking_yolo/blob/95477ded20c2dc8c69115b7de9063ec1da13204d/requirements.txt) dependencies installed, including `torch>=1.6`. In addition you will need to have setup Jupyter with CUDA support for GPU. 

## Running the project
Download the dataset N-Caltech101 from https://www.garrickorchard.com/datasets/n-caltech101 and unzip in the project root. 

With Jupyter, run [data_processing.ipynb](https://github.com/Zaabon/spiking_yolo/blob/95477ded20c2dc8c69115b7de9063ec1da13204d/data_processing.ipynb). This might take a while, you might not need to process all data. 

The full network can be run through [spiking_yolov3.ipynb](https://github.com/Zaabon/spiking_yolo/blob/95477ded20c2dc8c69115b7de9063ec1da13204d/spiking_yolov3.ipynb) while only the original YOLOv3 is runnable through [original_yolov3.ipynb](https://github.com/Zaabon/spiking_yolo/blob/95477ded20c2dc8c69115b7de9063ec1da13204d/original_yolov3.ipynb). 

Code for spiking solution is modified from https://github.com/yjwu17/BP-for-SpikingNN and can be found in the directory SpikingNN. 

Most of main YOLOv3 code can be found in [train.py](https://github.com/Zaabon/spiking_yolo/blob/95477ded20c2dc8c69115b7de9063ec1da13204d/train.py) which is originally from https://github.com/ultralytics/yolov3

