import os
import pandas as pd
import matplotlib.pyplot as plt

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

def store_data(data, file_path, file_name):
    with open(f"{file_path}/data/{file_name}.txt", "a") as f:
        f.write(str(data) + "\n")

def plot_data(file_path, file_name):
    datas = []
    with open(f"{file_path}/data/{file_name}.txt", "r") as file:
        for line in file:
            data = float(line.strip())
            datas.append(data)

    plt.plot(datas)
    plt.xlabel("Step")
    plt.ylabel(f"{file_name}")
    plt.title(f"{file_name}")
    plt.savefig(f"{file_path}/{file_name}")
    plt.close()
