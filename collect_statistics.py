import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
from tools.FaceDetection import VideoAnalyser
import time
import cv2
from utils.utils import parse_time_window
import seaborn as sns

global models_lib
models_lib = "/e/data/models"
config = {
    "match_condition_flag": True,
    "model_type": "resnet"
}

movie_path = "/e/data/movies_dictionary.csv"
movies = pd.read_csv(movie_path)
age_all, gender_all, participants_num = [], [], []
for i, movie in movies.iterrows():
    print(f"movie {i} / {len(movies)}")
    times_to_run = movie["times_to_run"][1:-1]
    video_path = movie.url
    detector = VideoAnalyser(video_path, models_lib=models_lib, config=config)  # types: dlib, resnet, yolo
    fps, frame_count = detector.get_movie_features()
    times_to_analyze = str.split(times_to_run, ",")
    for time_window in times_to_analyze:
        first_frame_time, last_frame_time = parse_time_window(time_window)
        age, gender = detector.get_frame_participants_data(first_frame_time * 1000)
        age_all += age
        gender_all += gender
        participants_num.append(len(age))
df= pd.DataFrame({"age": age_all, "gender": gender_all})
df["data"]=1
df["gender"] = df["gender"] .replace("Man", "Male")
df["gender"] = df["gender"] .replace("Woman", "Female")
labels, counts = np.unique(gender_all,return_counts=True)
ax = sns.violinplot(x="data", y="age", hue="gender", data=df, palette="Set2", split=True, scale="count")
plt.title(f"{labels[0]} : {counts[0]},  {labels[1]} : {counts[1]}")
plt.savefig("/e/data/results/final_graphs/age_gender_distribution")

participants_num_arr = np.array(participants_num)
participants_num_arr2 = participants_num_arr[participants_num_arr>=4]
sns.displot(pd.DataFrame({"participants_num": participants_num_arr2}), x="participants_num", discrete=True)
plt.title(f"participants number in meeting distribution")
plt.savefig("/e/data/results/final_graphs/articipants number in meeting distribution")