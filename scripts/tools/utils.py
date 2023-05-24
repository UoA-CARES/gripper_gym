import os
import pandas as pd
import matplotlib.pyplot as plt

def create_directories(local_results_path, folder_name):
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    file_path = f"{local_results_path}/{folder_name}"
    
    if not os.path.exists(f"{local_results_path}/{folder_name}"):
        os.makedirs(f"{local_results_path}/{folder_name}")
    if not os.path.exists("servo_errors"): #servo error still here because it's used by servo.py which shouldn't know the local storage
        os.makedirs("servo_errors")
    return file_path

def plot_curve(data, file_path, file_name):
    data = pd.DataFrame.from_dict(data)
    data.to_csv(f"{file_path}/{file_name}", index=False)
    data.plot(x='step', y=f'episode_{file_name}', title=file_name)
    plt.savefig(f"{file_path}/{file_name}")
    plt.close()