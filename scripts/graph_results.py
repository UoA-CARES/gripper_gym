import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
import plotly.graph_objects as go
import collections
import pathlib

Z = 1.960 # 95% confidence interval
X_VALS_FILE_NAME = "steps_per_episode.txt"
Y_VALS_FILE_NAME = "success_list.txt"
X_LIMIT = 30000

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder1",      type=str)
    parser.add_argument("--folder2",      type=str)
    parser.add_argument("--folder3",      type=str)
    parser.add_argument("--plot_type",      type=str)
    parser.add_argument("--title",      type=str)
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

def plot_single_graph(datas_map, title, window_size=50):
    plt.ioff()
    figure = plt.figure()
    figure.set_figwidth(6)
    sns.set_theme(style="darkgrid")

    for key in datas_map:
        x_label = "x"
        y_label = "y"

        df = pd.DataFrame({'x' : datas_map[key]['x'], 'y': datas_map[key]['y']})

        # confidence interval stuff
        df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
        movStd = df[y_label].rolling(window=window_size, min_periods=1).std()

        confIntPos = df["avg"] + Z * movStd / np.sqrt(window_size)
        confIntNeg = df["avg"] - Z * movStd / np.sqrt(window_size)

        ax = sns.lineplot(data=df, x=x_label, y="avg", label=key)

        # ax.set_xlim(1,12000) # Limit graph to specific x value

        ax.set(xlabel=x_label, ylabel="avg")
        plt.fill_between(df[x_label], confIntNeg, confIntPos, alpha=0.2)

    sns.move_legend(ax, "lower right")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.show()

def plot_algo_compare(datas_map_arr, titles, window_size=20):
    plt.ioff()
    sns.set()
    fig, axes = plt.subplots(1, len(datas_map_arr), constrained_layout=True)
    sns.set_theme(style="darkgrid")

    for i, datas_map in enumerate(datas_map_arr):
        for key in datas_map:
            x_label = "x"
            y_label = "y"

            df = pd.DataFrame({'x' : datas_map[key]['x'], 'y': datas_map[key]['y']})

            # confidence interval stuff
            df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
            movStd = df[y_label].rolling(window=window_size, min_periods=1).std()

            confIntPos = df["avg"] + Z * movStd / np.sqrt(window_size)
            confIntNeg = df["avg"] - Z * movStd / np.sqrt(window_size)

            axes[i] = sns.lineplot(ax=axes[i], data=df, x=x_label, y="avg", label=key)
            axes[i].set_xlim(1,X_LIMIT) # Limit graph to specific x value
            axes[i].fill_between(df[x_label], confIntNeg, confIntPos, alpha=0.2)
            
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Set x-axis tick labels in scientific notation

        sns.move_legend(axes[i], "lower right")

    axes[0].set_ylabel('Average Reward')
    plt.show()

def plot_success_rate(datas_map_arr, titles, window_size=100):
    plt.ioff()
    sns.set()
    fig, axes = plt.subplots(1, len(datas_map_arr), constrained_layout=True)
    sns.set_theme(style="darkgrid")

    for i, datas_map in enumerate(datas_map_arr):
        for key in datas_map:
            x_label = "x"
            y_label = "y"

            df = pd.DataFrame({'x' : datas_map[key]['x'], 'y': datas_map[key]['y']})

            # confidence interval stuff
            df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
            movStd = df[y_label].rolling(window=window_size, min_periods=1).std()

            confIntPos = df["avg"] + Z * movStd / np.sqrt(window_size)
            confIntNeg = df["avg"] - Z * movStd / np.sqrt(window_size)

            axes[i] = sns.lineplot(ax=axes[i], data=df, x=x_label, y="avg", label=key)
            axes[i].set_xlim(1,X_LIMIT) # Limit graph to specific x value
            axes[i].fill_between(df[x_label], confIntNeg, confIntPos, alpha=0.2)
            
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Set x-axis tick labels in scientific notation

        sns.move_legend(axes[i], "lower right")

    axes[0].set_ylabel('Success Rate')
    plt.show()

# def plot_G_values(folder1, folder2, folder3):
#     folders = []
#     task_to_algos = {}
#     titles = ["TD3 30-330", "SAC 30-330", "DDPG 30-330"]

#     parse_folder(folder1, folders)
#     parse_folder(folder2, folders)
#     parse_folder(folder3, folders)

#     for sub_dirs in folders:
#         datas_map = {}

#         for sub_dir in sub_dirs:
#             folder_name = os.path.basename(sub_dir)
#             splitted = folder_name.split("_")
#             algorithm = splitted[-2]

#             # Regex to extract G value at end
#             pattern = r"G\d+"
#             matches = re.findall(pattern, folder_name)

#             if matches:
#                 g_val = matches[0][1:]
#             else:
#                 g_val = "error"

#             step_file = f"{sub_dir}/data/{X_VALS_FILE_NAME}"
#             reward_file = f"{sub_dir}/data/{Y_VALS_FILE_NAME}"

#             key = algorithm + " G:" + g_val
#             datas_map[key] = { "x": [] , "y": []}
#             parse_step(step_file, datas_map[key]["x"])
#             parse_reward(reward_file, datas_map[key]["y"])

#         od = collections.OrderedDict(sorted(datas_map.items()))

#         task_name = os.path.basename(sub_dir)
#         task_to_algos[task_name] = od
    
#     print(type(task_to_algos))
#     plot_multiple(task_to_algos, titles)

def plot_training_evaluation(root_folder):
    dataframes = []
    titles = []

    for task_folder in pathlib.Path(root_folder).iterdir():
        if not task_folder.is_dir(): pass
        datas_map = {}
        for algo_folder in pathlib.Path(task_folder).iterdir():
            if not algo_folder.is_dir(): pass
            folder_name = os.path.basename(algo_folder)
            splitted = folder_name.split("_")
            algorithm = splitted[-1]

            csv_file = f"{algo_folder}/data/{folder_name}_evaluation"
            df = pd.read_csv(csv_file)

            datas_map[algorithm] = df

        od = collections.OrderedDict(sorted(datas_map.items()))
        dataframes.append(od)

        splitted = os.path.basename(task_folder).split("_")
        task = splitted[-1]
        titles.append(f"{task}")

    plot_evals(dataframes, titles)

def plot_evals(dataframes, titles, window_size=20):
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
            movStd = df[y_label].rolling(window=window_size, min_periods=1).std()

            confIntPos = df["avg"] + Z * movStd / np.sqrt(window_size)
            confIntNeg = df["avg"] - Z * movStd / np.sqrt(window_size)

            axes[i] = sns.lineplot(ax=axes[i], data=df, x=x_label, y="avg", label=key)
            axes[i].set_xlim(1,X_LIMIT) # Limit graph to specific x value
            axes[i].fill_between(df[x_label], confIntNeg, confIntPos, alpha=0.2)
            
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # Set x-axis tick labels in scientific notation

        sns.move_legend(axes[i], "lower right")

    axes[0].set_ylabel('Average Reward')
    plt.show()

def plot_different_algorithms(folder1, success_rate):
    datas_map_arr = []
    titles = []

    for task_folder in pathlib.Path(folder1).iterdir():
        if not task_folder.is_dir(): pass
        datas_map = {}
        for algo_folder in pathlib.Path(task_folder).iterdir():
            if not algo_folder.is_dir(): pass
            folder_name = os.path.basename(algo_folder)
            splitted = folder_name.split("_")
            algorithm = splitted[-1]

            step_file = f"{algo_folder}/data/{X_VALS_FILE_NAME}"
            reward_file = f"{algo_folder}/data/{Y_VALS_FILE_NAME}"

            datas_map[algorithm] = { "x": [] , "y": []}
            parse_step(step_file, datas_map[algorithm]["x"])
            parse_reward(reward_file, datas_map[algorithm]["y"])

        od = collections.OrderedDict(sorted(datas_map.items()))
        datas_map_arr.append(od)

        splitted = os.path.basename(task_folder).split("_")
        task = splitted[-1]
        titles.append(f"{task}")
    
    if (success_rate):
        plot_success_rate(datas_map_arr, titles)
    else:
        plot_algo_compare(datas_map_arr, titles)

def plot_single(folder1, title, plot_type):

    sub_dirs = [f.path for f in os.scandir(folder1) if f.is_dir()]
    datas_map = {}
    for sub_dir in sub_dirs:
        folder_name = os.path.basename(sub_dir)
        splitted = folder_name.split("_")
        algorithm = splitted[-2]

        if plot_type == "g_single":
            # Regex to extract G value at end
            pattern = r"G\d+"
            matches = re.findall(pattern, folder_name)

            if matches:
                g_val = matches[0][1:]
            else:
                g_val = "error"

            step_file = f"{sub_dir}/data/{X_VALS_FILE_NAME}"
            reward_file = f"{sub_dir}/data/{Y_VALS_FILE_NAME}"
            key = algorithm + " G:" + g_val
        else:
            key = algorithm

        step_file = f"{sub_dir}/data/{X_VALS_FILE_NAME}"
        reward_file = f"{sub_dir}/data/{Y_VALS_FILE_NAME}"

        datas_map[key] = { "x": [] , "y": []}
        parse_step(step_file, datas_map[key]["x"])
        parse_reward(reward_file, datas_map[key]["y"])

    od = collections.OrderedDict(sorted(datas_map.items()))
    
    plot_single_graph(od, title)

def plot_evaluation(folder1, title):
    # sub_dirs = [f.path for f in os.scandir(folder1) if f.is_dir()]
    datas_map = {}
    folder_name = os.path.basename(folder1)
    splitted = folder_name.split("_")
    algorithm = splitted[-2]

    step_file = f"{folder1}/data/{X_VALS_FILE_NAME}"
    reward_file = f"{folder1}/data/{Y_VALS_FILE_NAME}"

    key = algorithm

    datas_map[key] = { "x": [] , "y": []}
    parse_step(step_file, datas_map[key]["x"])
    parse_reward(reward_file, datas_map[key]["y"])

    od = collections.OrderedDict(sorted(datas_map.items()))
    
    plot_single_graph(od, title)

# plot graphs with desired comparison type
def main():
    args = parse_args()
    # if (args.plot_type == "g"):
    #     plot_G_values(args.folder1, args.folder2, args.folder3)
    if (args.plot_type == "algorithms"):
        plot_different_algorithms(args.folder1, success_rate=False)
    if (args.plot_type == "success"):
        plot_different_algorithms(args.folder1, success_rate=True)
    elif (args.plot_type == "g_single" or args.plot_type == "algorithms_single"):
        plot_single(args.folder1, args.title, args.plot_type)
    elif (args.plot_type == "post_training_evaluation"):
        plot_evaluation(args.folder1, args.title)
    elif (args.plot_type == "training_evaluation"):
        plot_training_evaluation(args.folder1)
    else:
        raise ValueError("Invalid plot type")

if __name__ == "__main__":
    main()
