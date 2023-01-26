import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from common import helper, instance

current_path = os.path.dirname(os.path.realpath(__file__))

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
VIEW_X_MIN = - 500
VIEW_X_MAX = 500
VIEW_Y_MIN = - 500
VIEW_Y_MAX = 500
VIEW_Z_MIN = 0
VIEW_Z_MAX = 1000

POSE_ANCHORS = [
    [0,1],
    [0,2],
    [1,3],
    [2,4],
    [5,6],
    [5,7],
    [6,8],
    [7,9],
    [8,10],
    [5,11],
    [6,12],
    [11,12],
    [11,13],
    [12,14],
    [13,15],
    [14,16]
]

def run():
    # json and video output path
    output_path = os.path.join(current_path, "output")
    # get json name
    json_name = [item for item in os.listdir(output_path) if item.split(".")[-1]=="json"]
    # get json full path
    EXTRACTED_DATA_PATH =  [os.path.join(output_path, item) for item in json_name]
    for file_infex, json_path in enumerate(EXTRACTED_DATA_PATH):
        video_name =  json_name[file_infex].split(".")[0]
        print("#"*10, "processing video:", video_name, "#"*10)
        # create video frame folder
        frames_path = os.path.join(current_path, "frames", video_name)
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)
            
        # Load json data
        extracted_data = helper.load_extracted_data(json_path)
        for index, data in enumerate(tqdm(extracted_data, desc="plot frames")):
            # set frame name
            frame_name = "{}.png".format(index)
            
            pose = instance.Pose.load(data.pose[0])
            # calibarte
            detections = [instance.Detection.load(detection) for detection in data.detection]
            # get human object detection information
            detection = instance.Detection.filter(detections, 0)
            if index == 0:
                BASELINE_HEIGHT = detection.height # get object height
                BASELINE_VERTICAL_OFFSET = detection.y_max - pose.y.max()
            # get pose data
            pose = helper.calibrate(data, FRAME_HEIGHT, BASELINE_HEIGHT, BASELINE_VERTICAL_OFFSET)
            if pose is None: # if no human poses are detected
                print("Frame {} didn't detect any human pose!".format(index+1))
                continue

            plt.style.use('dark_background')
            fig = plt.figure(figsize=(20, 12))
            plt.scatter(pose.x, pose.y, color="red")
            plt.xlim([VIEW_X_MIN, VIEW_X_MAX])
            plt.ylim([VIEW_Z_MIN, VIEW_Z_MAX])
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            for pose_anchor in POSE_ANCHORS:
                ax.plot(pose.x[pose_anchor], pose.y[pose_anchor], color="#ffffff", linewidth=5)
            plt.savefig(os.path.join(frames_path, frame_name))
            plt.close(fig)
            
        create_video(frames_path)
        # only plot one character

def create_video(frames_path, fps=25.0):
    frame_array = []
    output_path = os.path.join(current_path, "output")
    video_name = os.path.split(frames_path)[-1]

    for filename in os.listdir(frames_path):
        frame = cv2.imread(os.path.join(frames_path, filename))
        height, width, layers = frame.shape
        size = (width, height)
        frame_array.append(frame)
        
    out = cv2.VideoWriter(os.path.join(output_path, video_name + "_pose.mp4") ,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in tqdm(range(len(frame_array)), desc="saving video"):
        # writing to a image array
        out.write(frame_array[i])
    print("pose video has been saved to :", output_path)
    out.release()

if __name__ == "__main__":
    run()
