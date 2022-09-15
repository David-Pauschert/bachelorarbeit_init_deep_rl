import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle, islice


def percentile(n):
    """Calculate n-percentile

    Args:
        n: n as in n-percentile
    """
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def group_1d(df, algo: str, attr: str, perf_attr="performance", env=None):
    """Group a dataframe by the argument provided by attr and create mean and std for performance

    Args:
        df dataframe to be grouped
        algo (str): name of algorithm to be analyzed
        attr (str): name of attribute to base the first level of grouping on
        perf_attr (str, optional): attribute to base the performance evaluation on. Defaults to "performance".
        env (_type_, optional): name of environment to be analyzed. Defaults to None.

    Returns:
        grouped dataframe
    """
    df_red = df[(df["algo"] == algo) & (df["env"] == env)
                ] if env is not None else df[df["algo"] == algo]
    df_grouped = df_red.groupby([attr], as_index=False).agg(
        avg_perf=pd.NamedAgg(column=perf_attr, aggfunc="mean"),
        std_perf=pd.NamedAgg(column=perf_attr, aggfunc="std"),
        count=pd.NamedAgg(column="id", aggfunc="count")
    )
    return df_grouped

#  Group a dataframe by the two arguments provided by attr1 and attr2 and create mean and std for column "perf_attr"


def group2d(df, algo: str, attr1: str, attr2: str, perf_attr="performance", env: str = None):
    """Group a dataframe by the argument provided by attr and create mean and std for performance 

    Args:
        df dataframe to be grouped
        algo (str): name of algorithm to be analyzed
        attr1 (str): name of attribute to base the first level of grouping on
        attr2 (str): name of attribute to base the seconf level of grouping on
        perf_attr (str, optional): attribute to base the performance evaluation on. Defaults to "performance".
        env (_type_, optional): name of environment to be analyzed. Defaults to None.

    Returns:
        grouped dataframe
    """
    df_red = df[(df["algo"] == algo) & (df["env"] == env)
                ] if env is not None else df[df["algo"] == algo]
    df_grouped = df_red.groupby([attr1, attr2], as_index=False).agg(
        avg_perf=pd.NamedAgg(column=perf_attr, aggfunc="mean"),
        std_perf=pd.NamedAgg(column=perf_attr, aggfunc="std"),
        count=pd.NamedAgg(column="id", aggfunc="count")
    )
    return df_grouped


def plot_bar_1d_v2(labels, avg, title, std=None, y_label="Average Return P", save_path=None):
    """Create bar chart

    Args:
        labels: labels of bars
        avg: y-values
        title: title of bar chart
        std (optional): standard deviation of y-values. Defaults to None.
        y_label (str, optional): label of y-axis. Defaults to "Average Return P".
        save_path (optional): Path to the folder the plot is ought to be saved. Defaults to None.
    """
    colors = list(islice(cycle(['#33a8c7ff', '#52e3e1ff', '#a0e426ff', '#fdf148ff',
                  '#ffab00ff', '#f77976ff', '#f050aeff', '#d883ffff', '#9336fdff']), None, len(labels)))
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, avg, yerr=std, color=colors, capsize=10)
    plt.xticks(y_pos, labels, rotation=90)
    plt.title(title)
    plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()


def performance_quantile(df, algo: str, attr: str = None, env: str = None, q=0.1):
    """Get the rows of a dataframe that belong to the top n-percentile

    Args:
        df: dataframe
        algo (str): algorithm to be analyzed
        attr (str, optional): attribute on which the calculation is based. Defaults to None.
        env (str, optional): environment to be analyzed. Defaults to None.
        q (float, optional): percentile value 1. Defaults to 0.1.

    Returns:
       n-prtventile
    """
    best = df[(df["algo"] == algo) & (df["env"] == env)
              ] if env is not None else df[df["algo"] == algo]
    quantile_size = math.ceil(len(best) * q)
    best = best.nlargest(quantile_size, "performance")
    if attr:
        best = group_1d(best, algo, attr)
        best["share"] = best["count"] / quantile_size
    return best

# Create aggregate statistics for the collected data


def performance_table(df):
    """Calculate mean and std for performance value of every algorithm-environment combination

    Args:
        df: dataframe

    Returns:
        dataframe with aggregate statistics
    """
    table = df.groupby(["algo", "env"]).agg(
        max_perf=pd.NamedAgg(column="performance", aggfunc=max),
        quantile=pd.NamedAgg(column="performance", aggfunc=percentile(90))
    )
    return table
