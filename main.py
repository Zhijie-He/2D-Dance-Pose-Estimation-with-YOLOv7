import os
import torch
import gdown
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from torchvision import transforms
from common import helper, instance
from yolov7.utils.general import non_max_suppression_kpt, non_max_suppression
from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
import warnings
warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.realpath(__file__))
# parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

# Presetting
DETECTION_MODEL_WEIGHTS_PATH = os.path.join(current_path, "weights", "yolov7-e6e.pt")
POSE_MODEL_WEIGHTS_PATH = os.path.join(current_path, "weights", "yolov7-w6-pose.pt")

DETECTION_IMAGE_SIZE = 1920
POSE_IMAGE_SIZE = 960
STRIDE = 64
CONFIDENCE_TRESHOLD = 0.25
IOU_TRESHOLD = 0.65

# Pre-process
def detection_pre_process_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    img = letterbox(frame, DETECTION_IMAGE_SIZE, STRIDE, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def pose_pre_process_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    image = letterbox(frame, POSE_IMAGE_SIZE, stride=STRIDE, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    return image

# Post-process
def clip_coords(boxes: np.ndarray, img_shape: Tuple[int, int]):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1]) # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0]) # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1]) # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0]) # y2


def detection_post_process_output(
    output: torch.tensor, 
    confidence_trashold: float, 
    iou_trashold: float,
    image_size: Tuple[int, int],
    scaled_image_size: Tuple[int, int]
) -> np.ndarray:
    output = non_max_suppression(
        prediction=output,
        conf_thres=confidence_trashold,
        iou_thres=iou_trashold
    )
    coords = output[0].detach().cpu().numpy()
    
    v_gain = scaled_image_size[0] / image_size[0]
    h_gain = scaled_image_size[1] / image_size[1]

    coords[:, 0] /= h_gain
    coords[:, 1] /= v_gain
    coords[:, 2] /= h_gain
    coords[:, 3] /= v_gain

    clip_coords(coords, image_size)
    return coords


def post_process_pose(pose: np.ndarray, image_size: Tuple, scaled_image_size: Tuple) -> np.ndarray:
    height, width = image_size
    scaled_height, scaled_width = scaled_image_size
    vertical_factor = height / scaled_height
    horizontal_factor = width / scaled_width
    result = pose.copy()
    for i in range(17):
        result[i * 3] = horizontal_factor * result[i * 3]
        result[i * 3 + 1] = vertical_factor * result[i * 3 + 1]
    return result


def pose_post_process_output(
    output: torch.tensor, 
    confidence_trashold: float, 
    iou_trashold: float,
    image_size: Tuple[int, int],
    scaled_image_size: Tuple[int, int],
    pose_model
) -> np.ndarray:
    output = non_max_suppression_kpt(
        prediction=output, 
        conf_thres=confidence_trashold, 
        iou_thres=iou_trashold, 
        nc=pose_model.yaml['nc'], 
        nkpt=pose_model.yaml['nkpt'], 
        kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

        for idx in range(output.shape[0]):
            output[idx, 7:] = post_process_pose(
                output[idx, 7:], 
                image_size=image_size,
                scaled_image_size=scaled_image_size
            )

    return output


# Annotate
def detect_annotate(image: np.ndarray, detections: np.ndarray, color: instance.Color, thickness: int = 2) -> np.ndarray:
    annotated_image = image.copy()
    for x_min, y_min, x_max, y_max, confidence, class_id in detections:
        rect = instance.Rect(
            x=float(x_min),
            y=float(y_min),
            width=float(x_max - x_min),
            height=float(y_max - y_min)
        )
        annotated_image = helper.draw_rect(image=annotated_image, rect=rect, color=color, thickness=thickness)
    return annotated_image


def pose_annotate(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    annotated_frame = image.copy()

    for idx in range(detections.shape[0]):
        pose = detections[idx, 7:].T
        plot_skeleton_kpts(annotated_frame, pose, 3)

    return annotated_frame


def download_weights(yolov7_e6e_url, yolov7_w6_pose_url):
    # download weights
    weights_path = os.path.join(current_path, "weights")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    
    if not os.path.exists(os.path.join(weights_path, "yolov7-e6e.pt")):
        helper.download_file(yolov7_e6e_url, weights_path)

    if not os.path.exists(os.path.join(weights_path, "yolov7-w6-pose.pt")):
        helper.download_file(yolov7_w6_pose_url, weights_path)

def download_video_from_gdrive(cfg):
    # create video input path
    video_input_path = os.path.join(current_path, "input")
    output = os.path.join(video_input_path, cfg['video_name'])
    if not os.path.exists(video_input_path):
        os.makedirs(video_input_path)
        # get video absolute path
        gdown.download(id = cfg['test_video_google_download_id'], output = output, quiet=False)
    
    return [os.path.join(video_input_path, video) for video in os.listdir(video_input_path)]

def run(cfg):
    # download test video
    SOURCE_VIDEO_PATH = download_video_from_gdrive(cfg)
    # download yolov7 detection and pose estimation weights
    download_weights(cfg['yolov7_e6e_url'], cfg['yolov7_w6_pose_url'])
    
    # create output directory
    output_path = os.path.join(current_path, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # object detection model
    detection_model = attempt_load(weights=DETECTION_MODEL_WEIGHTS_PATH, map_location=cfg['device'])

    # human pose estimation model
    weigths = torch.load(POSE_MODEL_WEIGHTS_PATH, map_location=cfg['device'])
    pose_model = weigths["model"] # get pose models
    _ = pose_model.float().eval()
    if torch.cuda.is_available():
    # move pose_model to device
        pose_model.half().to(cfg['device'])
        
    for video_path in tqdm(SOURCE_VIDEO_PATH, desc="video"):
        # get video name
        video_name = os.path.split(video_path)[1].split(".")[0]
        print("#"*10, "processing video:", video_name, "#"*10)
        # output video path
        target_video_path = os.path.join(output_path, os.path.split(video_path)[1])
        target_json_path = os.path.join(output_path, video_name + ".json")
        # get the number of video frames
        video_frame_num = helper.get_frame_count(video_path)
        # initiate video writer
        video_config = instance.VideoConfig(fps=cfg['fps'], width=cfg['width'], height=cfg['height'])
        # create video writer
        video_writer = helper.get_video_writer(target_video_path=target_video_path,  video_config=video_config)
        # get frame iterator, convert video into frames
        frame_iterator = iter(helper.generate_frames(video_file=video_path))
        COLOR = instance.Color(r=0, g=0, b=255)

        # save json file
        entries = []
        
        # get the one of frames
        for frame in tqdm(frame_iterator, total=video_frame_num, desc="process frames"):
            annotated_frame = frame.copy()
            with torch.no_grad():
                # get image size
                image_size = frame.shape[:2]
                # detection
                detection_pre_processed_frame = detection_pre_process_frame(frame=frame, device=cfg['device'])
                detection_scaled_image_size = tuple(detection_pre_processed_frame.size())[2:]
                detection_output = detection_model(detection_pre_processed_frame)[0].detach().cpu()
                detection_output = detection_post_process_output(
                    output=detection_output,
                    confidence_trashold=CONFIDENCE_TRESHOLD,
                    iou_trashold=IOU_TRESHOLD,
                    image_size=image_size,
                    scaled_image_size=detection_scaled_image_size
                )
                annotated_frame = detect_annotate(image=annotated_frame, detections=detection_output, color=COLOR)

                # pose
                pose_pre_processed_frame = pose_pre_process_frame(frame=frame, device=cfg['device'])
                pose_scaled_image_size = tuple(pose_pre_processed_frame.size())[2:]

                pose_output = pose_model(pose_pre_processed_frame)[0].detach().cpu()
                pose_output = pose_post_process_output(
                    output=pose_output,
                    confidence_trashold=CONFIDENCE_TRESHOLD, 
                    iou_trashold=IOU_TRESHOLD,
                    image_size=image_size,
                    scaled_image_size=pose_scaled_image_size,
                    pose_model=pose_model
                )
                annotated_frame = pose_annotate(image=annotated_frame, detections=pose_output)

                # save video frame
                video_writer.write(annotated_frame)
                # save detection and pose output
                entry = {
                    # object detection - > two objects one ball and one human -> what if something else?
                    "detection": detection_output.tolist(),
                    # human pose detection
                    "pose": pose_output.tolist()
                }
                entries.append(entry)

        # close output video
        video_writer.release()
        # save json file
        helper.dump_json_file(file_path=target_json_path, content=entries)
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    configFilePath = os.path.join(current_path, "config/cfg.yaml")
    cfg = helper.load_config(configFilePath)
    cfg['device'] = opt.device
    # set device
    if opt.device == "gpu":
        cfg['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", cfg['device'])
        
    run(cfg)

