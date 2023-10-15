import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
import collections
import pathlib

Z = 1.960 # 95% confidence interval
X_VALS_FILE_NAME = "steps_per_episode.txt"
SUCCESS_FILE = "success_list.txt"
REWARD_FILE = "reward.txt"
X_LIMIT = 30000
CURRENT_DIR = os.getcwd()
GRAPHS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'graphs')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder_path",      type=str)
    parser.add_argument("--plot_type",      type=str)
    return parser.parse_args()

def parse_step(step_file, arr):
    curr_step = 0
    # parse step
    with open(step_file, "r") as file:
        for line in file:
            data = int(line.strip())
            curr_step += data
            arr.append(curr_step)

def parse_reward(reward_file, arr):
    # parse reward
    with open(reward_file, "r") as file:
        for line in file:
            data = float(line.strip())
            arr.append(data)

def parse_success_rate(success_file, arr):
    # parse success rate
    with open(success_file, "r") as file:
        for line in file:
            data = float(line.strip())
            arr.append(data)

def parse_folder(folder, folders_arr):
    if folder:
        sub_dirs = [f.path for f in os.scandir(folder) if f.is_dir()]
        folders_arr.append(sub_dirs)

def plot_reward(datas_map_arr, titles, window_size=20):
    """
    Plot rewards with confidence intervals.

    Args:
    datas_map_arr (list): List of data maps.
    titles (list): List of titles for subplots.
    window_size (int, optional): Window size for rolling mean and standard deviation. Defaults to 20.
    """
    plt.ioff()
    sns.set()
    fig, axes = plt.subplots(1, len(datas_map_arr), constrained_layout=True)
    sns.set_theme(style="darkgrid")

    for i, datas_map in enumerate(datas_map_arr):
        for key, data in datas_map.items():
            x_label = "x"
            y_label = "y"

            df = pd.DataFrame({'x': data['x'], 'y': data['y']})

            # Calculate confidence intervals
            df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
            mov_std = df[y_label].rolling(window=window_size, min_periods=1).std()

            conf_int_pos = df["avg"] + Z * mov_std / np.sqrt(window_size)
            conf_int_neg = df["avg"] - Z * mov_std / np.sqrt(window_size)

            # Plotting
            axes[i] = sns.lineplot(ax=axes[i], data=df, x=x_label, y="avg", label=key)
            axes[i].set_xlim(1, X_LIMIT)  # Limit graph to specific x value
            axes[i].fill_between(df[x_label], conf_int_neg, conf_int_pos, alpha=0.2)

        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Set x-axis tick labels in scientific notation

        sns.move_legend(axes[i], "lower right")

    axes[0].set_ylabel('Average Reward')
    fig.set_size_inches(16, 5)

    # Define the path for saving the figure
    figure_path = os.path.join(GRAPHS_DIR, 'reward_plot.png')
    plt.savefig(figure_path)  # Save the figure
    plt.show()

def plot_success_rate(datas_map_arr, titles, window_size=100):
    """
    Plot success rates with confidence intervals.

    Args:
    datas_map_arr (list): List of data maps.
    titles (list): List of titles for subplots.
    window_size (int, optional): Window size for rolling mean and standard deviation. Defaults to 100.
    """
    plt.ioff()
    sns.set()
    fig, axes = plt.subplots(1, len(datas_map_arr), constrained_layout=True)
    sns.set_theme(style="darkgrid")

    for i, datas_map in enumerate(datas_map_arr):
        for key, data in datas_map.items():
            x_label = "x"
            y_label = "y"

            df = pd.DataFrame({'x': data['x'], 'y': data['y']})

            # Calculate confidence intervals
            df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
            mov_std = df[y_label].rolling(window=window_size, min_periods=1).std()

            conf_int_pos = df["avg"] + Z * mov_std / np.sqrt(window_size)
            conf_int_neg = df["avg"] - Z * mov_std / np.sqrt(window_size)

            # Plotting
            axes[i] = sns.lineplot(ax=axes[i], data=df, x=x_label, y="avg", label=key)
            axes[i].set_xlim(1, X_LIMIT)  # Limit graph to specific x value
            axes[i].fill_between(df[x_label], conf_int_neg, conf_int_pos, alpha=0.2)

        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Set x-axis tick labels in scientific notation

        sns.move_legend(axes[i], "lower right")

    axes[0].set_ylabel('Success Rate')
    fig.set_size_inches(16, 5)

    # Define the path for saving the figure
    figure_path = os.path.join(GRAPHS_DIR, 'success_rate_plot.png')
    plt.savefig(figure_path)  # Save the figure
    plt.show()

def plot_training_evaluation(root_folder):
    """
    Plot training evaluation data from the given root folder.

    Args:
    root_folder (str): Root folder path.

    This function reads evaluation data from different algorithm folders within each task folder and plots the data accordingly.
    """
    dataframes = []
    titles = []

    # Sort the folders in numerical order
    sorted_task_folders = sorted(pathlib.Path(root_folder).iterdir())

    for task_folder in sorted_task_folders:
        if not task_folder.is_dir():
            continue

        datas_map = {}
        for algo_folder in pathlib.Path(task_folder).iterdir():
            if not algo_folder.is_dir():
                continue

            folder_name = os.path.basename(algo_folder)
            algorithm = folder_name.split("_")[-1]

            csv_file = f"{algo_folder}/data/{folder_name}_evaluation"
            df = pd.read_csv(csv_file)

            datas_map[algorithm] = df

        od = collections.OrderedDict(sorted(datas_map.items()))
        dataframes.append(od)

        task = os.path.basename(task_folder).split("_")[-1]
        titles.append(f"Rotation {task}")

    plot_evals(dataframes, titles)

def plot_evals(dataframes, titles, window_size=20):
    """
    Plot evaluation data.

    Args:
    dataframes (list): List of dataframes.
    titles (list): List of titles for subplots.
    window_size (int, optional): Window size for rolling mean and standard deviation. Defaults to 20.
    """
    plt.ioff()
    sns.set()
    fig, axes = plt.subplots(1, len(dataframes), constrained_layout=True)
    sns.set_theme(style="darkgrid")

    for i, datas_map in enumerate(dataframes):
        for key in datas_map:

            df = datas_map[key]
            x_label = "step"
            y_label = "avg_episode_reward"

            # confidence interval stuff
            df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
            mov_std = df[y_label].rolling(window=window_size, min_periods=1).std()

            conf_int_pos  = df["avg"] + Z * mov_std / np.sqrt(window_size)
            conf_int_neg  = df["avg"] - Z * mov_std / np.sqrt(window_size)

            axes[i] = sns.lineplot(ax=axes[i], data=df, x=x_label, y="avg", label=key)
            axes[i].set_xlim(1,X_LIMIT) # Limit graph to specific x value
            axes[i].fill_between(df[x_label], conf_int_neg, conf_int_pos, alpha=0.2)
            
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Set x-axis tick labels in scientific notation

        sns.move_legend(axes[i], "lower right")

    axes[0].set_ylabel('Average Reward')
    fig.set_size_inches(16, 5)

    # Define the path for saving the figure
    figure_path = os.path.join(GRAPHS_DIR, 'training_evaluation_plot.png')
    plt.savefig(figure_path)  # Save the figure
    plt.show()

def plot_different_algorithms(folder, success_rate):
    """
    Plot different algorithms based on the data in the provided folder.

    Args:
    folder (str): Path to the folder containing the data.
    success_rate (bool): Boolean indicating whether to plot success rates or rewards.

    This function reads data from different algorithm folders within each task folder and plots the data based on the type of plot specified.
    """
    datas_map_arr = []
    titles = []

    # Sort the folders in numerical order
    sorted_task_folders = sorted(pathlib.Path(folder).iterdir())

    for task_folder in sorted_task_folders:
        if not task_folder.is_dir():
            continue

        datas_map = {}
        for algo_folder in pathlib.Path(task_folder).iterdir():
            if not algo_folder.is_dir():
                continue

            folder_name = os.path.basename(algo_folder)
            algorithm = folder_name.split("_")[-1]

            step_file = f"{algo_folder}/data/{X_VALS_FILE_NAME}"
            if success_rate:
                vals_file = f"{algo_folder}/data/{SUCCESS_FILE}"
            else:
                vals_file = f"{algo_folder}/data/{REWARD_FILE}"

            datas_map[algorithm] = {"x": [], "y": []}
            parse_step(step_file, datas_map[algorithm]["x"])
            parse_reward(vals_file, datas_map[algorithm]["y"])

        od = collections.OrderedDict(sorted(datas_map.items()))
        datas_map_arr.append(od)

        task = os.path.basename(task_folder).split("_")[-1]
        titles.append(f"Rotation {task}")

    if success_rate:
        plot_success_rate(datas_map_arr, titles)
    else:
        plot_reward(datas_map_arr, titles)

def create_graphs_dir(graphs_dir):
    """
    Create the 'graphs' directory if it does not already exist.

    Args:
    graphs_dir (str): Directory path for saving graphs.
    """
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        print(f"Directory '{graphs_dir}' created successfully.")
    else:
        print(f"Directory '{graphs_dir}' already exists.")

# plot graphs with desired comparison type
def main():
    """
    Main function to execute the script for plotting graphs.
    """
    args = parse_args()
    graphs_dir = os.path.join(os.path.dirname(os.getcwd()), 'graphs')
    create_graphs_dir(graphs_dir)
    
    if args.plot_type == "reward":
        plot_different_algorithms(args.folder_path, success_rate=False)
    elif args.plot_type == "success_rate":
        plot_different_algorithms(args.folder_path, success_rate=True)
    elif args.plot_type == "training_evaluation":
        plot_training_evaluation(args.folder_path)
    else:
        raise ValueError("Invalid plot type")

if __name__ == "__main__":
    main()
