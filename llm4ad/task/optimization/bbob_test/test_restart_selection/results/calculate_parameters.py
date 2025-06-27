import re
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    folder_path_1 = [i + 1 for i in range(24)]
    x = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

    for i in folder_path_1:
        folder_path = [f"problem_{i}/Problem_{i}_Convergence_Run_{j}.txt" for j in range(31)]

        for j in x:
            avg_y = []

            for k in folder_path:
                curve = []

                with open(f"problem_{i}/Problem_{i}_Convergence_Run_{k}.txt", "r") as f:
                    txt_content = f.read()
                    pattern = r'np\.float64\((\d+\.\d+)\)'
                    numbers = re.findall(pattern, txt_content)
                    values = np.array([float(num) for num in numbers])

                    count = 0
                    for l in range(1, len(values)):
                        if values[l] - values[l-1] <= j:
                            count += 1
                        else:
                            curve.append([j, count])
                            count = 0
