import os
import matplotlib.pyplot as plt
from pathlib import Path

def create_directories(local_results_path, folder_name):
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    file_path = f"{local_results_path}/{folder_name}"
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(f"{file_path}/data"):
        os.makedirs(f"{file_path}/data")
    if not os.path.exists("servo_errors"): #servo error still here because it's used by servo.py which shouldn't know the local storage
        os.makedirs("servo_errors")
    return file_path

def store_configs(file_path, env_config, gripper_config, learning_config, object_config):
    with open(f"{file_path}/configs.txt", "w") as f:
        f.write(f"Environment Config:\n{env_config.json()}\n")
        f.write(f"Gripper Config:\n{gripper_config.json()}\n")
        f.write(f"Learning Config:\n{learning_config.json()}\n")
        f.write(f"Learning Config:\n{object_config.json()}\n")
        with open(Path(env_config.camera_matrix)) as cm:
            f.write(f"\nCamera Matrix:\n{cm.read()}\n")
        with open(Path(env_config.camera_distortion)) as cd:
            f.write(f"Camera Distortion:\n{cd.read()}\n")

def store_data(data, file_path, file_name):
    with open(f"{file_path}/data/{file_name}.txt", "a") as f:
        f.write(str(data) + "\n")

def plot_data(file_path, files):
    if type(files) is not list:
        files = [files]

    for file_name in files:
        datas = []
        with open(f"{file_path}/data/{file_name}.txt", "r") as file:
            for line in file:
                data = float(line.strip())
                datas.append(data)

        plt.plot(datas)
        plt.xlabel("Episode")
        plt.ylabel(f"{file_name}")
        plt.title(f"{file_name}")
        plt.savefig(f"{file_path}/{file_name}")
        plt.close()

def slack_post_plot(environment, slack_bot, file_path, plots):
    if type(plots) is not list:
        plots = [plots]

    for plot_name in plots:
        if os.path.exists(f"{file_path}/{plot_name}.png"):
            slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: {plot_name}", f"{file_path}/", f"{plot_name}.png")
        else:
            slack_bot.post_message("#cares-chat-bot", f"#{environment.gripper.gripper_id}: {plot_name} plot not ready yet or doesn't exist")
