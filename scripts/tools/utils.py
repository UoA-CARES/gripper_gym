import os
import shutil
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

def store_configs(file_path, parent_path, folder_name = "configs"):
    if not os.path.isdir(f"{file_path + '/' + folder_name}"):
        os.mkdir(file_path + '/' + folder_name)

    for file_name in os.listdir(parent_path):
    # construct full file path
        source = parent_path + "/" + file_name

        destination = file_path + "/" + folder_name + "/" + file_name
        print(f"Destination: {destination}")

        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)


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

def plot_data_time(file_path, files, file_name_average_reward, file_name_time):
    average_reward = []
    time = []
    if type(files) is not list:
        files = [files]

    with open(f"{file_path}/data/{file_name_average_reward}.txt", "r") as file:
        for line in file:
            data = float(line.strip())
            average_reward.append(data)

    with open(f"{file_path}/data/{file_name_time}.txt", "r") as file:
        for line in file:
            data = float(line.strip())
            time.append(data)            

        plt.plot(time, average_reward)
        plt.xlabel("Time")
        plt.ylabel(f"{file_name_average_reward}")
        plt.title("Average Reward vs Time")
        plt.savefig(f"{file_path}/reward_average_vs_time")
        plt.close()        

def slack_post_plot(environment, slack_bot, file_path, plots):
    if type(plots) is not list:
        plots = [plots]

    for plot_name in plots:
        if os.path.exists(f"{file_path}/{plot_name}.png"):
            slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: {plot_name}", f"{file_path}/", f"{plot_name}.png")
        else:
            slack_bot.post_message("#cares-chat-bot", f"#{environment.gripper.gripper_id}: {plot_name} plot not ready yet or doesn't exist")
