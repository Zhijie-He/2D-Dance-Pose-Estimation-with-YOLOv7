import os
import cv2
import json
import wget
import yaml
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from typing import Generator, List, Optional, Tuple
from .instance import Rect, Color, VideoConfig, FrameData, Pose, Detection, Point

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        
def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    # cv2 read video file
    video = cv2.VideoCapture(video_file)

    if (video.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame
    video.release()


def plot_image(image: np.ndarray, size: int = 12, image_name = "frame_example", output_path = os.path.join(parent_path, "output")) -> None:
    plt.figure(figsize=(size, size))
    plt.axis('off')
    plt.imshow(image[...,::-1])
    plt.savefig(os.path.join(output_path, image_name + ".png"))


def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image

def download_file(url, output_directory):
    filename = wget.download(url, output_directory)
    print("\n", filename, "has been downloaded!")


# create cv2.VideoWriter object that we can use to save output video
def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path, 
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
        fps=video_config.fps, 
        frameSize=(video_config.width, video_config.height), 
        isColor=True
    )


def get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def create_parent_dir(file_path: str) -> None:
    file_directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(file_directory, exist_ok=True)

def dump_json_file(file_path: str, content: Union[list, dict], **kwargs) -> None:
    create_parent_dir(file_path=file_path)
    with open(file_path, "w") as file:
        json.dump(content, file, **kwargs)


def load_json(path: str) -> dict:
    with open(path) as f:
        contents = f.read()
        return json.loads(contents)


def load_extracted_data(path: str) -> List[FrameData]:
    raw = load_json(path)
    return [
        FrameData(
            pose=entry['pose'],
            detection=entry['detection']
        )
        for entry
        in raw
    ]


def calibrate(
    data: FrameData, 
    frame_height: int, 
    baseline_pose_height: float,
    baseline_vertical_offset: float
) -> Optional[Tuple[Pose, Point]]:
    detections = [Detection.load(detection) for detection in data.detection]
    # get person detection information
    detection_person = Detection.filter(detections, 0)
    # get ball detection information -> 32 
    #detection_ball = Detection.filter(detections, 32)
    
    if detection_person is None:
        return None
    # if detection_ball is None:
    #     return None
    if len(data.pose) != 1:
        return None

    # ball_x, ball_y = detection_ball.center.int_xy_tuple

    pose = Pose.load(data.pose[0])
    pose.y = frame_height - pose.y
    #ball_y = frame_height - ball_y
 
    x_shift = (pose.x.max() + pose.x.min()) / 2
    y_shift = pose.y.min() - baseline_vertical_offset

    pose.x = pose.x - x_shift
    #ball_x = ball_x - x_shift
    pose.y = (pose.y - y_shift) * 1000 / baseline_pose_height
    #ball_y = (ball_y - y_shift) * 1000 / baseline_pose_height
    return pose

def load_config(configFilePath):
    with open(configFilePath, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg
