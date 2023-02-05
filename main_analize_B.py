import argparse
import os
import pickle
from pathlib import Path
from utils.utils import is_url
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import DotConfig
from utils.face_func import iou
from tools.FaceDetection import VideoAnalyser
import time
import cv2
import numpy as np

pd.set_option('display.width', 320, 'display.max_columns', 10)
global models_lib
models_lib = "/e/data/models"

def main():
    video_path = args.link
    run_name = video_path[-6:] if is_url(video_path) else Path(video_path.replace(" ", "_")).stem
    with open(args.config, 'r') as stream:
        config = DotConfig(yaml.safe_load(stream))
    full_out_name = os.path.join(config.output_dir, run_name)
    frames_out_dir = os.path.join(full_out_name, "frames")
    config.full_out_name = full_out_name
    config.frames_out_dir = frames_out_dir
    if not os.path.isdir(full_out_name):
        os.mkdir(full_out_name)
    if not os.path.isdir(frames_out_dir):
        os.mkdir(frames_out_dir)
    detector = VideoAnalyser(video_path, models_lib=models_lib, config=config)    # types: dlib, resnet, yolo
    fps, frame_count = detector.get_movie_features()
    if fps == 0 or frame_count == 0:
        raise ValueError("Movie cannot be loaded")
    time.sleep(1)
    movie_duration = frame_count / fps
    first_frame_time, last_frame_time = float(args.times.split("-")[0]) * 60, float(args.times.split("-")[1]) * 60
    last_frame_time = min(last_frame_time, movie_duration)
    time_steps_sec = np.arange(first_frame_time, last_frame_time, 1 / config.kps)
    start_anlyze_time = time.time()
    for i, frame_time in enumerate(time_steps_sec):
        if i % 30 == 0:
            print(f"frame number: {i} / {time_steps_sec[-1]}. video name: {video_path}")
        is_frame, frame= detector.get_frame_B(frame_time * 1000, save_frame=(i % 1 == 0))
        if not is_frame:
            print(f"{run_name} - problem with frame: {i}")
    print("analysis have been finish. wait to save results!")
    detector.save_face_dict()
    end_anlyze_time = time.time()
    print(f"analysis time: {end_anlyze_time - start_anlyze_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process youtube downloader.')
    parser.add_argument('link', help='youtube link or path to saved movie')
    parser.add_argument('config', help='config file')
    parser.add_argument('-k', '--kps', type=float, default=1, help='desired frames rate of the output images')
    parser.add_argument('-t', '--times', default="0-60", type=str, help='string of minutes to analyze for example: 0-10')
    args = parser.parse_args()

    plt.rcParams["figure.figsize"] = (20, 20)

    main()

# args example https://www.youtube.com/embed/48KjeTeWe7Y
# times for the https://www.youtube.com/embed/48KjeTeWe7Y movie (60 frames):
#   only retina locations: 20s
#   retina location plusface_recognition.face_encodings: 43s
#   retina location plus fer_model.predict_emotion_vector:  28s
#   retina locations plus DeepFace.analyze("emotion"):  29s

