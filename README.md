# 💃 2D Dance Pose Estimation with YOLOv7

<img src="images/result.gif" width=100% height=250>

## Abstract
This project uses YOLOv7 algorithm to detect object and estimate huamn pose. A test video has been given in the format of google file id. Detection and huamn pose data will be saved in the format of json and will be saved in the output folder.

If need to change another video, delete input folder videos and put new video under input folder. Note could solve multiple videos.

Note: For plot_json file, only one person video can be plot.

## Getting Started
Clone the repository.

```
git clone https://github.com/Zhijie-He/2D-Dance-Pose-Estimation-with-YOLOv7.git
```

Go to the cloned folder.
```
cd 2D-Dance-Pose-Estimation-with-YOLOv7
```
## Steps to run Code
Create virtual environment
```
### For Linux Users
python3 -m venv yolov7_dance
source yolov7_dance/bin/activate

### For Window Users
python3 -m venv yolov7_dance
cd yolov7_dance
cd Scripts
activate
cd ..
cd ..
```
### Installation
Install requirements with mentioned command below.
```
pip install -r requirements.txt
```

### Object detection and huamn pose estimation
In the configuration file (config/cfg.yaml), a test video has been gived which can be downloaded by gdown. 

Run the code
```
python main.py --device [gpu/cpu]
```
The object dection and huamn pose estimation data will be saved in output file with the json format.
<img src="images/object_human_pose_estimation.gif" width=100% height=250>

Plot human pose with json file using matplotlib
```
python plot_json.py
```
<img src="images/result.gif" width=100% height=250>
