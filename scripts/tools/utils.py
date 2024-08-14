import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import socket


def position_to_pixel(position, reference_position, camera_matrix):
    # pixel_n = f * N / Z + c_n
    pixel_x = (
        camera_matrix[0, 0]
        * (position[0] + reference_position[0])
        / reference_position[2]
        + camera_matrix[0, 2]
    )
    pixel_y = (
        camera_matrix[1, 1]
        * (position[1] + reference_position[1])
        / reference_position[2]
        + camera_matrix[1, 2]
    )
    return int(pixel_x), int(pixel_y)


def create_directories(local_results_path, folder_name):
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    file_path = f"{local_results_path}/{folder_name}"

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(f"{file_path}/data"):
        os.makedirs(f"{file_path}/data")
    if not os.path.exists(
        "servo_errors"
    ):  # servo error still here because it's used by servo.py which shouldn't know the local storage
        os.makedirs("servo_errors")
    return file_path


def store_configs(file_path, parent_path, folder_name="configs"):
    if not os.path.isdir(f"{file_path + '/' + folder_name}"):
        os.mkdir(file_path + "/" + folder_name)

    for file_name in os.listdir(parent_path):
        # construct full file path
        source = parent_path + "/" + file_name

        destination = file_path + "/" + folder_name + "/" + file_name
        print(f"Destination: {destination}")

        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print("copied", file_name)


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
            slack_bot.upload_file(
                "#cares-chat-bot",
                f"#{environment.gripper.gripper_id}: {plot_name}",
                f"{file_path}/",
                f"{plot_name}.png",
            )
        else:
            slack_bot.post_message(
                "#cares-chat-bot",
                f"#{environment.gripper.gripper_id}: {plot_name} plot not ready yet or doesn't exist",
            )

def lineseg_dists(p,a,b):
    # Calculate shortest distance between points p and line segments defined by each point in a to b

    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1,1)))

    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    
    return np.hypot(h,c)

def get_values(id, host='localhost', port=65432):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))
            client_socket.sendall(id.encode('utf-8'))
            data = client_socket.recv(1024).decode('utf-8')
            list = eval(data)
            return list
    except ConnectionRefusedError:
        return "Failed to connect to the server."

def euclidean_dist(p1,p2):

    temp = p1 - p2

    euclid_dist = np.sqrt(np.dot(temp.T, temp))

    return euclid_dist

def save_evaluation_values(data_eval_reward, filename, file_path):
    data = pd.DataFrame.from_dict(data_eval_reward)
    data.to_csv(f"{file_path}/data/{filename}_evaluation", index=False)
    data.plot(x="step", y="avg_episode_reward", title="Evaluation Reward Curve")
    plt.savefig(f"{file_path}/data/{filename}_evaluation.png")
    plt.close()
