import os

import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# from cde import *

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the folder where you want to read/write files
folder_path = os.path.join(current_dir)

def run_single(method):

    # Initialization
    problem_number = 24  # Choose a problem instance range from f1 to f24
    run_number = 31
    if method != "METADE":
        all_data = {
            'Problem': [],
            'BestFoundResult': [],
            'Std': [],
            'AcceptanceReachPoint': []
        }
    else:
        all_data = {
            'Problem': [],
            'BestFoundResult': [],
            'Std': []
        }

    for i in range(1, problem_number + 1):
        problem_index = i
        result_path = f'../{method}/results/problem_{problem_index}/'

        for j in range(1, problem_number + 1):
            resp = f'../{method}/results/problem_{j}/'
            merge_all_runs(resp, problem_index, run_number)
            print(f"Results for Problem {problem_index} merged.")

        if 1 <= problem_index <= 24:
            filename = os.path.join('GNBG', f'f{problem_index}.mat')
            GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
            MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
            AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
            Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
            CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
            MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
            MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
            CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
            CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
            CompH = np.array(GNBG_tmp['Component_H'][0, 0])
            Mu = np.array(GNBG_tmp['Mu'][0, 0])
            Omega = np.array(GNBG_tmp['Omega'][0, 0])
            Lambda = np.array(GNBG_tmp['lambda'][0, 0])
            RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
            OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
            OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
        else:
            raise ValueError('ProblemIndex must be between 1 and 24.')

        if not os.path.exists(os.path.join(result_path, f"Problem_{problem_index}_Results.xlsx")):
            # print(f"Problem {problem_index} don't exist, skipping.")
            continue

        with open(os.path.join(result_path, f"Problem_{problem_index}_Results.xlsx"), 'rb') as f:
            data = pd.read_excel(f)
            # change every value in BestFoundResult in data to BestFoundResult - OptimumValue
            data['BestFoundResult'] = abs(data['BestFoundResult'] - OptimumValue)
            # get mean of both BestFoundResult and AcceptanceReachPoint, skip None, if no satisfy, inf
            best_found_result = data['BestFoundResult'].mean(skipna=True)
            std = data['BestFoundResult'].std(skipna=True)
            if method != "METADE":
                acceptance_reach_point = data['AcceptanceReachPoint'].mean(skipna=True)
            if pd.isna(best_found_result):
                best_found_result = float('inf')
            if method != "METADE":
                if pd.isna(acceptance_reach_point):
                    acceptance_reach_point = float('inf')
            # save to all_data
            all_data['Problem'].append(problem_index)
            all_data['BestFoundResult'].append(best_found_result)
            all_data['Std'].append(std)
            if method != "METADE":
                all_data['AcceptanceReachPoint'].append(acceptance_reach_point)

        # save all data to a new excel file named DE result
        all_data_df = pd.DataFrame(all_data)
        all_data_df.to_excel(os.path.join(f'../{method}/results/', f'{method}_result.xlsx'), index=False)
        print(f"Problem {problem_index} processed. BestFoundResult: {best_found_result}")

    print(f"All data saved to {method}_result.xlsx")

def merge_all_runs(result_path, problem_index, total_runs=31):
    # 合并所有运行结果
    all_dfs = []
    for run in range(total_runs):
        file_path = os.path.join(result_path, f"Problem_{problem_index}_Run_{run}.xlsx")
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            all_dfs.append(df)
            # 可选：删除单独的文件
            # os.remove(file_path)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_path = os.path.join(result_path, f"Problem_{problem_index}_Results.xlsx")
        final_df.to_excel(final_path, index=False)


if __name__ == "__main__":
    run_single('test_restart_selection')
    # run_single('L-SHADE')


