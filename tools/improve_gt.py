import os
import numpy as np
import pandas as pd
import re

pd.set_option('display.width', 320, 'display.max_columns', 20)

gt_file_path = "/e/data/movies_dataset.1.csv"
improved_file_path = "/e/data/movies_dataset_improved_embedding_fer_vec_Arima_False_th2_1.12.csv"
results_dir = "/e/data/results_1.12/anomalies_log"
# benchmark =  ["pred_fer_vec", "anomalies_win_2"]
# benchmark = ["face_encodings_Arima_False", "threshold_1.5", "anomalies_win_7"]
benchmark = ["embedding_fer_vec_Arima_False", "threshold_2", "anomalies_win_7"]  # run_type, ma_threshold, anomaly_window, face_ratio
gt = pd.read_csv(gt_file_path)
# gt["name"] = gt["url"].apply(lambda x: str(x)[-6:])
gt["name_orginal"] = [("_".join(video_name.split(" "))) for video_name in gt.video_name.values]
gt["name"] = gt["name_orginal"].apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
gt_new = gt.copy()
results_files = os.listdir(results_dir)
original_pickels_dir = "/e/data/thessis_movies/analysis_B"
data_libs = os.listdir(original_pickels_dir)
run_names = list(set([directory.split("/")[-1] for directory in data_libs]))

for run_name in run_names:
    relevant_results = [file for file in results_files if np.all(list(map(lambda x: x in file, [run_name] + benchmark)))]
    anomaly_events = "["
    last_up = -1
    for file in relevant_results:
        pred = pd.read_csv(os.path.join(results_dir, file))
        for _, row in pred.iterrows():
            new_last = int(np.ceil(row['end anomaly']))
            if abs(last_up - int(row['start anomaly'])) <= 3:
                anomaly_events = anomaly_events.replace(str(last_up), str(new_last))
            else:
                anomaly_events += f"{int(row['start anomaly'])}-{new_last}, "
            last_up = new_last

    anomaly_events = anomaly_events[:-1] + "]"  # TODO: cleaned repetitive windows
    anomaly_events = "[]" if anomaly_events == "]" else anomaly_events
    anomaly_events = anomaly_events.replace(",]", "]")
    gt_new.loc[gt_new.name.apply(lambda x: x.startswith(re.sub('[^a-zA-Z0-9 \n\.]', '', run_name)[:30])), "anomaly_events"] = anomaly_events
    print(gt_new.loc[gt_new.name.apply(lambda x: x.startswith(re.sub('[^a-zA-Z0-9 \n\.]', '', run_name)[:30])), "anomaly_events"])
    1

gt_new.to_csv(improved_file_path)
