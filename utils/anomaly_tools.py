from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import arima, auto_arima

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)


def find_anomalies_in_face(dfpiv, window=10, anomaly_threshold=1.8, start_p=5, max_p=20, use_ARIMA=False):
    union_frames_anomalies = 1
    dfpiv["frame_sec"] = dfpiv["frame_time"] / 1000
    dfpiv.fillna(method='ffill', inplace=True)
    faces = [i for i in dfpiv.columns if type(i) == int]
    df_anomalies = dfpiv.drop(["active"], axis=1).copy()
    first_run = 1
    print(f"start fin anomalies window: {window}, anomaly_threshold: {anomaly_threshold}, Arima: {use_ARIMA}")
    for face_ind, column2analyze in enumerate(faces):
        y_train2 = dfpiv[[column2analyze]].squeeze().copy().interpolate(method = 'nearest')
        y_train2.iloc[0] = y_train2.iloc[1] if np.isnan(y_train2.iloc[0]) else y_train2.iloc[0]
        # y_train2.dropna(inplace=True)
        if use_ARIMA: ## arima predictoruse_ARIMA
            # forecaster = auto_arima(y_train2.values, start_p=start_p, max_p=max_p, trace=True)
            # in_sample_preds = forecaster.predict_in_sample()
            # in_sample_preds = np.append(in_sample_preds[1:], in_sample_preds[-1])
            start_analyze_ind = np.argmin(y_train2.isna())
            y_train2 = y_train2[start_analyze_ind:]
            train_size = 15
            if len(y_train2) <= train_size or np.sum(np.isnan(y_train2[train_size-3: train_size])) > 0:
                df_anomalies[column2analyze] = 0
                continue
            train, test = y_train2[:15], y_train2[15:]
            history = [x for x in train]
            predictions = list()
            predict = list()
            if first_run:
                stepwise_model = auto_arima(train, start_p=1, start_q=2,
                                            max_p=2, max_q=3, m=7,
                                            start_P=1, seasonal=True,
                                            d=1, D=1, trace=True,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True)
            for ind, t in enumerate(test.index):
                # model = sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
                stepwise_model.fit(history)
                output = stepwise_model.predict(n_periods=1)
                predict.append(output[0])
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
                if ind % 200 == 0:
                    print(f"fit in process {ind}/{len(test.index)}. face id: {face_ind}/{len(faces)}")

            df_temp = pd.DataFrame({"actuals": y_train2.values, "predicted": train.values.tolist() + predictions})
            # df_temp = detect_classify_anomalies(df_temp, window=window, anomaly_threshhold=anomaly_threshold, use_ARIMA=use_ARIMA)
            df_temp = simple_anomaly_classify(df_temp, window=window, anomaly_threshold=anomaly_threshold, union_frames_anomalies=union_frames_anomalies, use_ARIMA=True)
            start_index_alloc = int(len(df_anomalies) != len(df_temp))
            df_anomalies.loc[start_analyze_ind + start_index_alloc - 1:, column2analyze] = df_temp["anomaly_points"].values
            first_run = 0

        else:  #simple predictor
            df_temp = pd.DataFrame({"actuals": y_train2.values})

            df_temp = simple_anomaly_classify(df_temp, window=window, anomaly_threshold=anomaly_threshold, union_frames_anomalies=union_frames_anomalies, use_ARIMA=False)
            # dfpiv[f"anomalies_simple_{column2analyze}"] = df_temp["anomaly_points"].values
            # df_anomalies.loc[1:, column2analyze] = df_temp["anomaly_points"].values
            df_anomalies[column2analyze] = df_temp["anomaly_points"].values

    return df_anomalies


def detect_classify_anomalies(df , window, anomaly_threshhold=2, use_ARIMA=False):
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0,inplace=True)
    if use_ARIMA:
        df["error"]=0
        df['percentage_change']=0
        df['error']=df['actuals']-df['predicted']
        df['percentage_change'] = np.abs((df['actuals'] - df['actuals'].mean()- df['predicted'] + df['predicted'].mean()))/ (df['actuals']) * 100
        df.fillna(0, inplace=True)
        df['meanval'] = df['error'].rolling(window=window).mean()
        df['deviation'] = df['error'].rolling(window=window).std()
        df['anomaly_points'] = df['percentage_change'] > 4
        df['anomaly_points'] = np.where(df['anomaly_points'], df['actuals'], np.nan)
        return df
    else:
        df['meanval'] = df['actuals'].rolling(window=window).mean()
        df["error"] = df['actuals'] - df['meanval']
        df['deviation'] = df['actuals'].rolling(window=window).std()
        df['-3s'] = df['meanval'] - (2 * df['deviation'])
        df['3s'] = df['meanval'] + (2 * df['deviation'])
        df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
        df['2s'] = df['meanval'] + (1.75 * df['deviation'])
        df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
        df['1s'] = df['meanval'] + (1.5 * df['deviation'])
        cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
        cut_values = cut_list.values
        cut_sort = np.sort(cut_values)
        df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                                   range(len(df['error']))]
        severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
        region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE", 6: "POSITIVE",
                  7: "POSITIVE"}
        df['color'] =  df['impact'].map(severity)
        df['region'] = df['impact'].map(region)
        df['anomaly_points'] = np.where(df['color'] >= anomaly_threshhold, df['error'], np.nan)

        return df


def simple_anomaly_classify(df, window=5, anomaly_threshold=2, union_frames_anomalies = 9, use_ARIMA=False):
    """

    :param df:
    :param window: frames window size for moving average
    :param anomaly_threshhold:
    :param union_frames_anomalies: numper of frame in which several anomalies are considered as one (and the strongest is taken)
    :return:
    """
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    if use_ARIMA:
        df["error"]=0
        df['error']=df['actuals']-df['predicted']
        df['meanval'] = df['error'].rolling(window=window).mean()
        df['deviation'] = df['error'].rolling(window=window).std()
        df['down_dev'] = df['predicted'] - df['meanval'] - (anomaly_threshold * df['deviation'])
        df['up_dev'] = df['predicted'] + df['meanval'] + (anomaly_threshold * df['deviation'])

    else:
        df['actuals'].iloc[0] = df['actuals'].iloc[1]
        df['meanval'] = df['actuals'].rolling(window=window).mean()
        df["error"] = df['actuals'] - df['meanval']
        df['deviation'] = df['actuals'].rolling(window=window).std()
        df['down_dev'] = df['meanval'] - (anomaly_threshold * df['deviation'])
        df['up_dev'] = df['meanval'] + (anomaly_threshold * df['deviation'])

    df['down_dev'] = np.where(df['actuals'] == 0, np.nan, df['down_dev'])
    df['up_dev'] = np.where(df['actuals'] == 0, np.nan, df['up_dev'])
    mask_up = df["actuals"] > df["up_dev"]
    mask_down = df["actuals"] < df["down_dev"]
    df['anomaly_points'] = np.where(mask_up|mask_down, df['actuals'], np.nan)
    # delete consecutive anomalies point and take the one with highest value
    df['anomaly_points'].fillna(0, inplace=True)
    k = union_frames_anomalies
    if k>1:
        df["k_anomaly"] = df['anomaly_points'].rolling(k).max().shift(-k+int(k/2))
        df["anomaly_points"] = np.where(df['anomaly_points']==df["k_anomaly"], df["k_anomaly"], np.nan)
        df["anomaly_points"] = np.where(df["anomaly_points"] > 0, df["anomaly_points"], np.nan)
    return df


def plot_anomalies_graph(df,ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(14, 8))
    ax.plot(df.index, df.actuals, color="b", label = "actual")
    if "predicted" in df.columns:
        ax.plot(df.index, df.predicted, color="g", linestyle="-", linewidth=2, label="prediction")
    else:
        ax.plot(df.index, df.meanval, color="r", linestyle="-",linewidth=2, label="rolling value")
        ax.plot(df.index, df.down_dev, color="r", linestyle="--", label="rolling +/- deviation")
        ax.plot(df.index, df.up_dev, color="r", linestyle="--")
    ax.scatter(df.index, df.anomaly_points, color="r", s=40, marker="o", label="anomalies")
    ax.legend()
    ax.xlabel("frame index")
    return fig, ax


def map_to_minimal_number(active_number):
    if active_number < 7:
        return 3
    elif active_number < 11:
        return 4
    elif active_number < 14:
        return 5
    elif active_number < 17:
        return 6
    elif active_number < 20:
        return 7
    elif active_number <= 24:
        return 8


def simple_clustering(dfpiv, df_anomalies):

    df_anomalies.replace(0, np.nan, inplace=True)
    dfpiv["minimal_participants"] = dfpiv["active"].apply(lambda x:map_to_minimal_number(x))
    df_anomalies["general_anomaly"] = df_anomalies["num_anomalies_in_area"].values > dfpiv["minimal_participants"]
    df_anomalies["cluster_start"] = df_anomalies["general_anomaly"] - df_anomalies["general_anomaly"].shift(1) == 1
    df_anomalies["cluster_end"] = df_anomalies["general_anomaly"] - df_anomalies["general_anomaly"].shift(-1) == 1
    start_cluster_array = np.where(df_anomalies["cluster_start"])[0]
    end_cluster_array = np.where(df_anomalies["cluster_end"])[0]
    if len(start_cluster_array) == len(end_cluster_array):
        pass
    elif len(start_cluster_array) == len(end_cluster_array) + 1:
        end_cluster_array = np.append(end_cluster_array, len(df_anomalies)-1)
    else:
        raise ValueError('WTF')

    num_participants = dfpiv["active"].values[start_cluster_array]
    faces= [i for i in dfpiv.columns if type(i) == int]
    if len(faces) == 0:
        return start_cluster_array, end_cluster_array, num_participants, None, None
    fig, ax = plt.subplots(len(faces), figsize=(12, 8))
    for i, f in enumerate(faces):
        ax[i].plot(dfpiv["frame_sec"], dfpiv[f], color='blue', label='actual')
        ax[i].scatter(dfpiv["frame_sec"], df_anomalies[f], marker="o", s=12, color='red', label='anomaly point')
        ax[i].set_title(f"face {f}")
        ax[i].set_xlim([dfpiv.iloc[0]["frame_sec"], dfpiv.iloc[-1]["frame_sec"]])

        if f != faces[-1]:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        else:
            ax[i].set_xlabel("video time [sec]")
        # draw anomalies area
        for j, (start_ind, end_ind) in enumerate(zip(start_cluster_array, end_cluster_array)):
            ax[i].axvspan(df_anomalies["frame_sec"][start_ind], df_anomalies["frame_sec"][end_ind], alpha=0.4, color='red')
    start_cluster_times = df_anomalies["frame_sec"].iloc[start_cluster_array].values
    end_cluster_times = df_anomalies["frame_sec"].iloc[end_cluster_array].values
    return start_cluster_times, end_cluster_times, num_participants,  fig, ax


def find_anomalies_by_expression(df, expression_columns=['pred_deepf', 'pred_fer_vec'], change_rolling_period=5):
    """
    for each column in the dataframe, return df tells if more than half of the expressions in the [rolling_window] have been changed
    """
    df.reset_index(inplace=True)
    df = df.drop_duplicates(['frame_time', 'face_index'])
    dc = df[:].pivot(index='frame_time', columns='face_index', values=expression_columns)
    minimal_num_appearance = 20
    dc = dc.loc[:, dc.isna().sum(axis=0) < len(dc) - minimal_num_appearance]
    if dc.shape[0] < 5 or dc.shape[1] < 4:
        return None,None
    long_disapear_threshold = 20  # how many frame where participant x is not detected is considered as long time to ignore him and remain with the None. optherwise the
    mask_long_nan = dc.rolling(long_disapear_threshold, min_periods=0).count() > 0
    dc.fillna(method='ffill', inplace=True)
    dc.where(mask_long_nan.values, np.nan, inplace=True) # for short disapearance we just interpulate the value. long disapearanve are remained Nones.
    active_faces = dc[expression_columns[0]].count(axis=1)
    dc.replace(np.nan, 0, inplace=True)
    dc_change = dc.shift(1) != dc  # pd.DataFrame(data=np.zeros_like(dc), columns = dc.columns)
    dc_roll = dc_change.rolling(change_rolling_period).median()
    dc_roll.replace(np.nan, 0, inplace=True)
    dc_roll["active"] = active_faces
    return dc_change.astype(int), dc_roll
