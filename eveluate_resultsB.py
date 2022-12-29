import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
import numpy as np
import re
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from utils.plots import plot_results, print_segmented_results

pd.set_option('display.width', 320, 'display.max_columns', 20)


def parse_ytrue_from_windows(gt_str, array_size):
    """
    return y_true list with ones in anomaly tome windows
    """
    y_true = [0] * array_size
    if gt_str:
        gt_list = str.split(gt_str, ",")
        min2sec_changer = lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]) if ":" in t else int(t)
        annomalies_win_gt = [(min2sec_changer(time_window.split("-")[0]), min2sec_changer(time_window.split("-")[1])) for time_window in gt_list]
        for win in annomalies_win_gt:
            max_ind = min(win[1]+1, array_size)
            y_true[win[0]: max_ind: 1] = [1] * (max_ind - win[0])
    return y_true


def find_run_type(name):
    run_types = ["face_encodings", "embedding_fer_vec", "embedding_deepf_vec", "pred_deepf", "pred_fer_vec"]
    for run_type in run_types:
        if run_type in name:
            return run_type
    return name


def parse_ypred_from_results(pred, array_size):
    y_pred = [0] * array_size
    for _, row in pred.iterrows():
        y_pred[int(row["start anomaly"]): int(np.ceil(row["end anomaly"]))+1: 1] = [1] * (int(np.ceil(row["end anomaly"]))+1-int(row["start anomaly"]))
    return y_pred

def main():
    gt = pd.read_csv(args.g_file)
    gt.dropna(subset=["url"], inplace=True)
    gt["name_orginal"] = [("_".join(video_name.split(" "))) for video_name in gt.video_name.values]
    gt["name"] = gt["name_orginal"].apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
    results_files = os.listdir(args.results)
    files_done = []
    regex_changed_part = re.compile('(?:\d*\.\d+|\d+)_to_(?:\d*\.\d+|\d+)se')
    regex_threshold = re.compile('_threshold_(\d*\.\d+|\d+)_')
    regex_window = re.compile('_anomalies_win_(\d*\.\d+|\d+)_')
    regex_face_ratio = re.compile('_faces_ratio_(\d*\.\d+|\d+)')
    log_run_type, log_threshold, log_anomaly_window, log_run_name, log_arima, log_face_ratio, log_num_prticipants, log_no_anomalies, log_confusion_matrix = [], [], [], [], [], [], [], [], []
    for i, file in enumerate(results_files):
        if i % 250 == 0:
            print(f"run {i} / {len(results_files)}")
        if file in files_done:
            continue
        changed_part  = regex_changed_part.findall(file)
        if changed_part:
            first_part = file.split(changed_part[0])[0]
            second_part = file.split(changed_part[0])[1]
        else:
            first_part = file
            second_part= ""
        files_with_same_name = [f for f in results_files if first_part in f and second_part in f]
        names_to_file = np.array([re.sub('[^a-zA-Z0-9 \n\.]', '', file).startswith(video_name.replace(".", "")) for video_name in gt["name"].values])
        if not np.any(names_to_file):
            raise ValueError("wrong gt is run")
        run_name = gt["name_orginal"].values[names_to_file][0]
        if gt.loc[gt.name_orginal == run_name, "anomaly_events"] is None:
            print(f"{run_name}: writs empty [] in the GT['anomalt_time'] ")
        current_gt_str = gt.loc[gt.name_orginal == run_name, "anomaly_events"].iloc[0][1:-1]
        duration_parts = gt.loc[gt.name_orginal == run_name]["duration"].values[0].split(":")
        y_size = int(duration_parts[2]) + int(duration_parts[1])*60 + int(duration_parts[0])*3600 + 1
        y_true = parse_ytrue_from_windows(current_gt_str, y_size)
        y_pred = [0] * y_size
        num_prticipants = gt.loc[gt.name_orginal == run_name]["num_participants"].iloc[0]
        for file in files_with_same_name:
            pred = pd.read_csv(os.path.join(args.results, file))
            for _, row in pred.iterrows():
                up_ind = min(int(np.ceil(row["end anomaly"])) + 1, y_size)
                y_pred[int(row["start anomaly"]): up_ind: 1] = [1] * (up_ind - int(row["start anomaly"]))
        if len(np.unique(y_true))>2 or len(np.unique(y_pred))>2:
            1
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        log_num_prticipants.append(num_prticipants)
        log_run_type.append(find_run_type(first_part))
        log_threshold.append(-1 if not regex_threshold.findall(file) else float(regex_threshold.findall(file)[0]))
        log_anomaly_window.append(int(regex_window.findall(file)[0]))
        log_run_name.append(run_name)
        log_arima.append("Arima_True" in file)
        log_no_anomalies.append(len(current_gt_str) == 0)
        log_confusion_matrix.append(cm)
        files_done += files_with_same_name
    results_df = pd.DataFrame({"run_name": log_run_name,"run_type": log_run_type,"num_participants": log_num_prticipants,  "ma_threshold":log_threshold,"anomaly_window":log_anomaly_window,
                               "use_Arima": log_arima, "no_anomalies": log_no_anomalies,  "confusion_matrix": log_confusion_matrix})
    results_df["recall"] = results_df["confusion_matrix"].apply(lambda cm: cm[1,1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[0, 1]) != 0 else np.nan)
    results_df["precision"] = results_df["confusion_matrix"].apply(lambda cm: cm[1,1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) != 0 else np.nan)
    results_df["specificity "] = results_df["confusion_matrix"].apply(lambda cm: cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else np.nan)
    results_df["FPR "] = results_df["confusion_matrix"].apply(lambda cm: cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else np.nan)
    results_df["F1"] = 2 * results_df["recall"] * results_df["precision"] / (results_df["recall"] + results_df["precision"])
    plot_results(results_df, args.save_dir)
    print_segmented_results(results_df, args.save_dir)
    1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='avalute results according to parametrs.')
    parser.add_argument("-r", "--results", default="/e/data/results_1.12/anomalies_log", help='directory to results')
    parser.add_argument("-g", "--gt-file", default="/e/data/movies_dataset.1.csv", help='ground true file path')
    parser.add_argument("-s", "--save-dir", default="/e/data/results_B/final_graphs", help='directory to save final graphs')
    args = parser.parse_args()
    main()