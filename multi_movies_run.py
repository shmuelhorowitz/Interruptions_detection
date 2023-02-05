import subprocess
import os
import pandas as pd
import time
from utils.utils import is_url
import re
pd.set_option('display.width', 320, 'display.max_columns', 20)


def isnan(x):
    return x != x


movie_path = "/e/data/movies_dictionary.csv"
overwrite_data_flag = True  # if true: overwrite existing directories

save_freq = 1

save_results_dir = "/e/data/thessis_movies/analysis_B"
if not os.path.isdir(save_results_dir):
    os.mkdir(save_results_dir)
already_analised = os.listdir(os.path.join(save_results_dir))
movies = pd.read_csv(movie_path)
movies_dir = "/e/data/video/"
movies_list = [m for m in os.listdir(movies_dir) if m.endswith("mp4")]
movies_to_remove = []
for movie in movies_list:
    try:
        if movie in movies_to_remove:
            continue
        if not overwrite_data_flag and os.path.isdir(os.path.join(save_results_dir, movie)):
            continue
        start_time = time.time()
        cmd = ["python", "main_analize_B.py", f"'{os.path.join(movies_dir, movie)}'", "/e/Dev/Thessis/config.yaml"]
        print(" ".join(cmd))
        subprocess.call(cmd)
        print(f'time for {movie} is: {round(time.time()-start_time,1)} sec')
    except:
        print(f"problen in movie: {movie}")
        continue


