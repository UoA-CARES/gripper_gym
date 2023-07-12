import os
import re
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import plotly.graph_objects as go

from pathlib import Path
file_path = Path(__file__).parent.resolve()

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--root_folder",      type=str)
    parser.add_argument("--plot_type",      type=str)

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

def parse_step(step_file, step_map, key):
    curr_step = 0
    # parse step
    with open(step_file, "r") as file:
        for line in file:
            data = int(line.strip())
            curr_step += data
            step_map[key].append(curr_step)

def parse_reward(reward_file, reward_map, key):
    # parse reward
    with open(reward_file, "r") as file:
        for line in file:
            data = float(line.strip())
            reward_map[key].append(data)

def plot_G_values(root_folder):
    reward_map = {}
    step_map = {}

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
            print(g_val)  # Output: G5
        else:
            g_val = "error"
            print("No match found.")

        step_file = f"{sub_dir}/data/steps_per_episode.txt"
        reward_file = f"{sub_dir}/data/rolling_reward_average.txt"

        reward_map[g_val] = []
        step_map[g_val] = []
        
        parse_step(step_file, step_map, g_val)
        parse_reward(reward_file, reward_map, g_val)

    datas = []
    for key in reward_map:
        print(key)
        data = go.Line(x=step_map[key], y=reward_map[key], name=f"{algorithm} G:{key}")
        datas.append(data)

    title = f"G effect on {algorithm} training"
    fig = create_fig(title, datas)
    fig.show()
    save_fig(fig, title)
    

def plot_different_algorithms(root_folder):
    reward_map = {}
    step_map = {}

    sub_dirs = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    base_folder_name = os.path.basename(root_folder)
    task = base_folder_name.split(" ")[0] # task name should be first word in space separated array

    for sub_dir in sub_dirs:
        folder_name = os.path.basename(sub_dir)
        splitted = folder_name.split("_")
        algorithm = splitted[-2]

        step_file = f"{sub_dir}/data/steps_per_episode.txt"
        reward_file = f"{sub_dir}/data/rolling_reward_average.txt"

        reward_map[algorithm] = []
        step_map[algorithm] = []
        
        parse_step(step_file, step_map, algorithm)
        parse_reward(reward_file, reward_map, algorithm)

    datas = []
    for key in reward_map:
        data = go.Line(x=step_map[key], y=reward_map[key], name=key)
        datas.append(data)

    title = f"Reward vs Step for task {task} degrees"
    fig = create_fig(title, datas)
    fig.show()
    save_fig(fig, title)

# Example of how to use Gripper
def main():
    args = parse_args()
    if (args.plot_type == "g"):
        plot_G_values(args.root_folder)
    elif (args.plot_type == "algorithm"):
        plot_different_algorithms(args.root_folder)
    else:
        raise ValueError("Invalid plot type")

if __name__ == "__main__":
    main()
