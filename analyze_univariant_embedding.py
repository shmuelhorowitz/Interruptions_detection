import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import arima
from pmdarima import model_selection
from pmdarima import pipeline
from pmdarima import preprocessing
from utils.anomaly_tools import find_anomalies_in_face, plot_anomalies_graph, simple_clustering, find_anomalies_by_expression

pd.set_option('display.width', 320, 'display.max_columns', 10)
plt.ioff()

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
one_face_threshold = 0.65
all_face_threshold = 0.8
percentage_of_faces = 0.51
response_time = 2000  # [ms]
response_time_frames = 4  # int(response_time / frames_delay)


def find_anomalied_arime(X: pd.DataFrame, y: pd.Series):
    date_feat = preprocessing.DateFeaturizer(
        column_name="time",  # the name of the date feature in the X matrix
        with_day_of_week=False,
        with_day_of_month=False)
    pipe = pipeline.Pipeline([
        ('date', date_feat),
        ('arima', arima.AutoARIMA(d=0,
                                  trace=3,
                                  stepwise=True,
                                  suppress_warnings=True,
                                  seasonal=False))
    ])
    pipe.fit(y, X)
    in_sample_preds, in_sample_confint = pipe.predict_in_sample(X, return_conf_int=True)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, y, color='blue', label='Training Data')
    ax.plot(X, in_sample_preds, color='green', marker='.',label='Predicted')
    ax.plot(X, in_sample_confint[:, 1], color='red', label='upper bound')
    ax.plot(X, in_sample_confint[:, 0], color='red', label='lower bound')
    ax.legend(loc='lower left', borderaxespad=0.5)
    plt.show()

def find_anomaly(*all_dist):
    topk_mean = np.mean(sorted(all_dist, reverse=True)[:int(np.ceil(np.count_nonzero(all_dist) * percentage_of_faces))])
    return topk_mean


def calc_topk_change(dfpiv, response_time_frames=4):
    droll = dfpiv.rolling(window=response_time_frames).max()
    droll.fillna(0, inplace=True)
    droll['topk_mean'] = droll[droll.columns.to_list()].apply(lambda x: find_anomaly(*x), axis=1)
    return droll['topk_mean']


def plt_rolling_frames(droll, fig_name):
    plt.figure(figsize=(12,
                        10))  # Two column paper. Each column is about 3.15 inch wide.
    sns.set_style("whitegrid")
    sns.color_palette()
    fig = plt.gcf()
    plt.rc('font', size=20)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=20)  # fontsize of the legend
    plt.ylim([0.5, 1])
    # fig.set_size_inches(15,12)
    b = sns.lineplot(x='frame_time', y="vals", hue='cols', data=droll, linewidth=3)
    # b.axes.set_title("Title",fontsize=50)
    b.set_xlabel("time [sec]")
    b.set_ylabel("mean change")
    plt.setp(b.get_legend().get_texts(), fontsize='22')  # for legend text
    plt.setp(b.get_legend().get_title(), fontsize='1')  # for legend titleb.tick_params(labelsize=5)
    plt.title(fig_name + ' \n \n meeting disruption index [MDI] for different response time', fontsize=22)
    plt.show()
    plt.savefig('graphs/accumulated change sensitivity to response time {}.jpg'.format(fig_name))


def analyze_time_series(data, run_name="", save_results=False, save_figs=False):
    window_for_anomalies = args.window_anomalies_size  # size of window to chcek and declare anomalies
    anomaly_threshold = args.anomaly_threshold  # threshold to detectc anomaly with ARIMA or with rolling average , for example 1.8
    epsilon = args.epsilon # frame distances between anomlaies in different faces that we  combine to one cluster of anomaly
     # frame distances between anomlaies in different faces that we  combine to one cluster of anomaly
    minimal_number_apperence = 10 # participants with less detections of this number are drpped out
    long_disapear_threshold = 10  # how many frame where participant I is not detected is considered as long tome enought time to ignore him.
    data = {k: v for k, v in data.items() if len(v) > 0}
    df = pd.DataFrame(data)
    frames = df.frame_time.unique()
    frames_delay = frames[1] - frames[0]
    df.sort_values(by=['face_index', 'frame_time'], ignore_index=True, inplace=True)
    df = df.set_index('face_index')
    df['frame_shifted'] = df.groupby(level=0)['frame_time'].shift(1)
    # df['is_relevant'] = df.apply(lambda x: (x['frame_time'] - x['frame_shifted']) == frames_delay, axis=1)
    df['embedding_fer_vec'] = df["embedding_fer_vec"].apply(lambda x: np.array(x))
    columns_to_analyzes = ["face_encodings", "embedding_fer_vec"] # , "embedding_deepf_vec"]

    for column_to_analyze in columns_to_analyzes:
        # if df[column_to_analyze].ndim==1:
        #     df[column_to_analyze] = df[column_to_analyze].apply(lambda x: np.expand_dims(x, axis=0))
        # df[column_to_analyze] = df[column_to_analyze].apply(lambda x: np.array(x) / sum(x))
        df[column_to_analyze] = df[column_to_analyze].apply(lambda y: np.nan if len(y) == 0 else y)
        df['analyzed_diff'] = df[column_to_analyze].diff()
        # df['delta_fer'] = df.apply(lambda row: np.linalg.norm(row[column_to_analyze] - row['analyzed_diff']) if row['is_relevant'] else np.NaN, axis=1)
        df['delta_fer'] = df['analyzed_diff'].apply(lambda row: np.linalg.norm(row))
        # df = df.drop(columns=[column_to_analyze, 'analyzed_diff'])
        df.reset_index(inplace=True)
        df = df.drop_duplicates(['delta_fer', 'face_index'])

        # analyse by delta fer vector
        dfpiv = df[:].pivot_table(index='frame_time', columns='face_index', values='delta_fer')
        # dfpiv = df[:].pivot(index='frame_time', columns='face_index', values='delta_fer')
        dfpiv = dfpiv.loc[:, dfpiv.isna().sum(axis=0) < len(dfpiv) - minimal_number_apperence]   #drop faces with small number of apperance

        # builde such that different component of participants will be separated
        mask_long_nan = dfpiv.rolling(long_disapear_threshold, min_periods=0).count() > 0
        dfpiv.where(mask_long_nan.values, dfpiv.interpolate(), inplace=True)  # for short disappearance we just interpolate the value. long disappearance are remained Nones.
        dfpiv.fillna(method='ffill', inplace=True)
        dfpiv["active"] = dfpiv.count(axis=1)  # count how many participants are active i each frame
        dfpiv.reset_index(inplace=True)
        # dfpiv_arima_anomalies = find_anomalies_in_face(dfpiv, use_ARIMA=True)
        for ARIMA_flag in [True, False]:
            df_anomalies = find_anomalies_in_face(dfpiv, window=window_for_anomalies, anomaly_threshold=anomaly_threshold, use_ARIMA=ARIMA_flag)
            df_anomalies.replace(0,np.nan,inplace=True)
            scene_name = f"{args.directory.split('/')[-1]}_{column_to_analyze}_Arima_{ARIMA_flag}"
            faces_ind = [i for i in dfpiv.columns if type(i) == int]
            colors = cm.rainbow(np.linspace(0, 1, len(faces_ind)))
            if 1:
                for ind, face_id in enumerate(faces_ind):
                    plt.plot(df_anomalies.frame_sec, dfpiv[face_id].values, color=colors[ind, :], label = face_id)
                    plt.scatter(df_anomalies.frame_sec, df_anomalies[face_id].values, color=colors[ind, :])
                plt.title(f"all faces changes. video: {scene_name} times {dfpiv.frame_sec.iloc[0]:.1f} - {dfpiv.frame_sec.iloc[-1]:.1f}sec. window:{window_for_anomalies}, thresholf:{anomaly_threshold}")
                plt.close()

            ## find clusters of anomalies - simple dbscan
            df_anomalies.replace(np.nan, 0, inplace=True)
            temp_anom = df_anomalies[faces_ind].where(df_anomalies[faces_ind]==0,1).rolling(epsilon).max()
            df_anomalies["num_anomalies_in_area"] = temp_anom.sum(axis=1)  # count how many faces with anomaly are detected in range of epsilon
            start_cluster_array, end_cluster_array, num_participants, fig, ax = simple_clustering(dfpiv, df_anomalies)
            if save_results:
                results_df = pd.DataFrame({"start anomaly": start_cluster_array, "end anomaly": end_cluster_array, "num_participants": num_participants})
                results_df.to_csv(os.path.join(args.output, "anomalies_log", f"{scene_name}_{run_name}_anomalies_win_{window_for_anomalies}_anomaly_threshold_{anomaly_threshold}_epsilon_{epsilon}.csv"))

            if fig is not None and save_figs:
                ax[0].set_title(f"video: {scene_name} times {dfpiv.frame_sec.iloc[0]:.1f} - {dfpiv.frame_sec.iloc[-1]:.1f}sec. window:{window_for_anomalies}, thresholf:{anomaly_threshold}")
                fig_path = os.path.join(args.output, "graphs", f"{scene_name}_{run_name}_anomalies_win_{window_for_anomalies}_anomaly_threshold_{anomaly_threshold}_epsilon_{epsilon}.png")
                plt.savefig(fig_path)
                plt.close()


    ## find annomalies by expressions
    expression_columns = ['pred_deepf', 'pred_fer_vec']
    dc_change, expressions_changes = find_anomalies_by_expression(df, expression_columns=expression_columns, change_rolling_period=window_for_anomalies)
    if dc_change is None:
        return
    expressions_changes.replace(np.nan, 0, inplace=True)
    print(f"analyze by pred dir: {args.directory.split('/')[-1]}")
    for exp_type in expression_columns:
        temp_anom = expressions_changes[exp_type]  #  .rolling(epsilon).max()
        dc_temp = dc_change[exp_type].copy()
        dc_temp["frame_sec"] = dc_change.index / 1000
        dc_temp["active"] = expressions_changes["active"]
        expressions_changes_temp = expressions_changes[exp_type].copy()
        expressions_changes_temp[f"num_anomalies_in_area"] = temp_anom.sum(axis=1)  # count how many faces with anomaly are detected in range of epsilon
        expressions_changes_temp["frame_sec"] = expressions_changes_temp.index / 1000
        expressions_changes_temp.reset_index(inplace=True)
        scene_name = f"{args.directory.split('/')[-1]}_{exp_type}"
        start_cluster_array, end_cluster_array, num_participants, fig, ax = simple_clustering(dc_temp, expressions_changes_temp)

        if save_results:
            results_df = pd.DataFrame({"start anomaly": start_cluster_array, "end anomaly": end_cluster_array,"num_participants": num_participants})
            results_df.to_csv(os.path.join(args.output, "anomalies_log",f"{scene_name}_{run_name}_anomalies_win_{window_for_anomalies}_anomaly_threshold_{anomaly_threshold}_epsilon_{epsilon}.csv"))
        if fig is not None and save_figs:
            ax[0].set_title(f"video: {scene_name} times {dc_temp.frame_sec.iloc[0]:.1f} - {dc_temp.frame_sec.iloc[-1]:.1f}sec. window:{window_for_anomalies}")
            fig_path = os.path.join(args.output, "graphs", f"{scene_name}_{run_name}_anomalies_win_{window_for_anomalies}_anomaly_threshold_{anomaly_threshold}_epsilon_{epsilon}.png")
            plt.savefig(fig_path)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse embbeding data frame.')
    parser.add_argument('directory', help='directory to analyse')
    parser.add_argument('-o', '--output', default="/e/data/results_B", help='path to results lib')
    parser.add_argument('-w', '--window-anomalies-size', default=5, type=int, help='siz eof window to detect deviation in')
    parser.add_argument('-a', '--anomaly-threshold', default=1.5, type=float, help='deviation threshold for anomaloius sample for ARIMA and rolling average methods')
    parser.add_argument('-e', '--epsilon', default=5, type=int, help='DBSCAN epsilon to connect samples to one cluster of anomalous')
    parser.add_argument('-s', '--save-fig', default=False, type=bool, help='DBSCAN epsilon to connect samples to one cluster of anomalous')
    parser.add_argument('-p', '--pred-run', default=False, type=bool, help='if true check only the predicition mode')

    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
        os.mkdir(os.path.join(args.output, "graphs"))
        os.mkdir(os.path.join(args.output, "anomalies_log"))

    plt.rcParams["figure.figsize"] = (20, 20)
    pickle_files = [f for f in os.listdir(args.directory) if f.endswith(".pkl")][:1]
    for i, file in enumerate(pickle_files):
        data = pd.read_pickle(os.path.join(args.directory, file))
        print(f"**** start analyzing dir: {args.directory}, file: {file}. {i+1}/{len(pickle_files)}")
        analyze_time_series(data, run_name = file[18:-4], save_results=True, save_figs=args.save_fig)
    print(f" args: name-{args.directory.split('/')[-1]}, window - {args.window_anomalies_size}, threshold- {args.anomaly_threshold}, epsilon- {args.epsilon}")
    anomaly_threshold = args.anomaly_threshold  # threshold to detectc anomaly with ARIMA or with rolling average , for example 1.8
    epsilon = args.epsilon

# run arguments example: /e/data/thessis_movies/analysis/eNOMJY -w 12 -a 1.4 -e 3
# /e/data/thessis_movies/analysis_B/Virtual_Panel_Discussion_The_Path_to_More_Flexible_AI -w 12 -a 1.4 -e 3