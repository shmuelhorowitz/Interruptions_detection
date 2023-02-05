import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np

def plot_results(df):
    sns.displot(data=df, x="precision", y="recall")

def print_segmented_results(df):
    grouped = df.groupby(["run_type",  "ma_threshold", "anomaly_window", "use_Arima"]).mean()  # "num_participants",
    grouped.reset_index()
    run_types = df.run_type.unique()
    for run_type in run_types:
        grouped = df.loc[df.run_type == run_type].dropna(subset=['recall' , 'precision']).groupby(["ma_threshold", "anomaly_window", "use_Arima"])\
            .agg('mean').reset_index()
        grouped = grouped.sort_values(by=['F1'], ascending=False)
        print(run_type)
        print(grouped.head(8))


def plot_minimal_number_of_participants():
    from utils.anomaly_tools import map_to_minimal_number
    import seaborn as sns
    x = range(4, 25)
    y = list(map(map_to_minimal_number, x))
    df = pd.DataFrame({"number of participants": x, "# participants with anomaly event": y})
    sns.lineplot(data=df, x="number of participants", y="# participants with anomaly event")
    g.set_xticklabels(np.linspace(4, 24, 4))
    g.set_xticks(np.linspace(4, 24, 6))
    sns.set_style('darkgrid')
    plt.show()