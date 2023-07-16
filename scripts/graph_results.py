import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
import plotly.graph_objects as go
import collections

Z = 1.960 # 95% confidence interval
X_VALS_FILE_NAME = "steps_per_episode.txt"
Y_VALS_FILE_NAME = "reward.txt"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root_folder",      type=str)
    parser.add_argument("--plot_type",      type=str)
    parser.add_argument("--title",      type=str)
    return parser.parse_args()

def save_fig(fig, title):
    home_path = os.path.expanduser('~')
    result_images_path = f"{home_path}/gripper_result_plots"
    if not os.path.exists(result_images_path):
        os.mkdir(result_images_path)

    fig.write_image(f"{result_images_path}/{title}.png")

def create_fig(title, datas):
    fig = go.Figure(
        data = datas,
        layout = {"xaxis": {"title": "Steps"}, "yaxis": {"title": "Average Reward"}, "title": title}
    )

    fig.update_layout(
        title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(
            family="Time New Roman",
            size=18,
        )
    )

    return fig

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

def plot_average(datas_map, window_size=50):
    plt.ioff()
    
    figure = plt.figure()
    figure.set_figwidth(6)
     
    sns.set_theme(style="darkgrid")

    for key in datas_map:
        # if key == "SAC":
        x_label = "x"
        y_label = "y"

        df = pd.DataFrame({'x' : datas_map[key]['x'], 'y': datas_map[key]['y']})

        # confidence interval stuff
        df["avg"] = df[y_label].rolling(window=window_size, min_periods=1).mean()
        movStd = df[y_label].rolling(window=window_size, min_periods=1).std()

        confIntPos = df["avg"] + Z * movStd / np.sqrt(window_size)
        confIntNeg = df["avg"] - Z * movStd / np.sqrt(window_size)

        ax = sns.lineplot(data=df, x=x_label, y="avg", label=key)

        ax.set_xlim(1,12000) # Limit graph to specific x value

        ax.set(xlabel=x_label, ylabel="avg")
        plt.fill_between(df[x_label], confIntNeg, confIntPos, alpha=0.2)

    sns.move_legend(ax, "lower right")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("90,180,270 Degrees Valve Rotation")
    plt.show()


def plot_G_values(root_folder, title):
    datas_map = {}

    sub_dirs = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    for sub_dir in sub_dirs:
        folder_name = os.path.basename(sub_dir)
        splitted = folder_name.split("_")
        algorithm = splitted[-2]

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
        datas_map[key] = { "x": [] , "y": []}
        parse_step(step_file, datas_map[key]["x"])
        parse_reward(reward_file, datas_map[key]["y"])

    od = collections.OrderedDict(sorted(datas_map.items()))
    plot_average(od)

def plot_different_algorithms(root_folder, title):
    datas_map = {}

    sub_dirs = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    base_folder_name = os.path.basename(root_folder)
    task = base_folder_name.split(" ")[0] # task name should be first word in space separated array

    for sub_dir in sub_dirs:
        folder_name = os.path.basename(sub_dir)
        splitted = folder_name.split("_")
        algorithm = splitted[-2]

        step_file = f"{sub_dir}/data/{X_VALS_FILE_NAME}"
        reward_file = f"{sub_dir}/data/{Y_VALS_FILE_NAME}"

        datas_map[algorithm] = { "x": [] , "y": []}
        parse_step(step_file, datas_map[algorithm]["x"])
        parse_reward(reward_file, datas_map[algorithm]["y"])

    od = collections.OrderedDict(sorted(datas_map.items()))
    plot_average(od)

# plot graphs with desired comparison type
def main():
    args = parse_args()
    if (args.plot_type == "g"):
        plot_G_values(args.root_folder, args.title)
    elif (args.plot_type == "algorithm"):
        plot_different_algorithms(args.root_folder, args.title)
    else:
        raise ValueError("Invalid plot type")

if __name__ == "__main__":
    main()
