import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


def analyze_early_stopping(b_values, num_problems=24, num_runs=31):
    """
    分析不同改进阈值b下，每个问题所需的早停计数a

    参数:
    b_values - 要测试的改进阈值列表
    num_problems - 问题数量
    num_runs - 每个问题的运行次数

    返回:
    results - 字典，键为b值，值为每个问题所需的早停计数列表
    """
    # 存储结果
    results = defaultdict(list)

    # 对每个b值进行分析
    for b in b_values:
        problem_min_counts = []

        for problem_id in range(1, num_problems + 1):
            run_min_counts = []

            for run_id in range(num_runs):
                print(f"正在处理问题 {problem_id} 的运行 {run_id + 1}，改进阈值 b = {b:.6f}")
                # 构建文件路径
                file_path = f"problem_{problem_id}/Problem_{problem_id}_Convergence_Run_{run_id}.txt"

                try:
                    # 读取收敛曲线数据
                    with open(file_path, "r") as f:
                        txt_content = f.read()
                        pattern = r'np\.float64\((\d+\.\d+)\)'
                        numbers = re.findall(pattern, txt_content)
                        values = np.array([float(num) for num in numbers])

                    # 找到连续改进小于b的最大次数
                    first_consecutive = 0
                    current_consecutive = 0

                    for i in range(1, len(values)):
                        # 计算相邻迭代的改进量
                        improvement = values[i] - values[i - 1]  # 注意：根据您的数据，可能需要调整这里的计算方式

                        if abs(improvement) <= b:  # 改进小于阈值b
                            current_consecutive += 1
                        else:
                            break

                    run_min_counts.append(first_consecutive)

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    continue

            if run_min_counts:
                # 对于该问题，取所有运行中的平均值作为安全的早停计数
                problem_min_counts.append(np.mean(run_min_counts))
            else:
                print(f"警告: 问题 {problem_id} 没有有效的运行数据")

        # 对于该b值，记录所有问题中需要的早停计数
        results[b] = problem_min_counts

    return results


def plot_results(results, b_values):
    """
    绘制分析结果的各种图表

    参数:
    results - 分析结果字典
    b_values - 改进阈值列表
    """
    # 计算每个b值下，所有问题所需的最小早停计数a
    min_a_for_all_problems = {b: max(counts) if counts else 0 for b, counts in results.items()}
    print("每个b值下所有问题所需的最小早停计数a:")
    for b, a in sorted(min_a_for_all_problems.items()):
        print(f"b = {b:.6f}, 最小a = {a}")

    # 绘制箱线图，展示每个b值下不同问题所需的早停计数分布
    plt.figure(figsize=(12, 6))
    box_data = [results[b] for b in b_values]
    plt.boxplot(box_data, labels=[f"{b:.6f}" for b in b_values])
    plt.xlabel('改进阈值b')
    plt.ylabel('所需早停计数a')
    plt.title('不同改进阈值b下各问题所需的早停计数a分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    # 在每个箱线图上标出最大值
    for i, b in enumerate(b_values):
        if results[b]:
            max_a = max(results[b])
            plt.text(i + 1, max_a, str(max_a), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('boxplot_early_stopping.png')
    plt.close()

    # 绘制散点图，展示b和a的关系
    plt.figure(figsize=(10, 6))
    b_values_sorted = sorted(b_values)
    a_values = [min_a_for_all_problems[b] for b in b_values_sorted]
    plt.plot(b_values_sorted, a_values, 'o-', markersize=8)
    plt.xscale('log')
    plt.xlabel('改进阈值b (对数刻度)')
    plt.ylabel('所需早停计数a')
    plt.title('改进阈值b与所需早停计数a的关系')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 标注点
    for b, a in zip(b_values_sorted, a_values):
        plt.text(b, a, f"({b}, {a})", ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig('scatter_early_stopping.png')
    plt.close()

    # 分析每个问题在不同b值下的早停计数变化
    plt.figure(figsize=(14, 8))
    for problem_id in range(len(results[b_values[0]])):
        problem_data = [results[b][problem_id] if problem_id < len(results[b]) else 0 for b in b_values_sorted]
        plt.plot(b_values_sorted, problem_data, '-', marker='o', label=f'问题{problem_id + 1}')

    plt.xscale('log')
    plt.xlabel('改进阈值b (对数刻度)')
    plt.ylabel('所需早停计数a')
    plt.title('各问题在不同改进阈值b下所需的早停计数a')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('problem_lines_early_stopping.png')
    plt.close()

    # 绘制热力图，展示不同问题和b值的组合
    plt.figure(figsize=(14, 10))
    num_problems = len(results[b_values[0]])
    heat_data = np.zeros((num_problems, len(b_values_sorted)))

    for i, b in enumerate(b_values_sorted):
        for p in range(min(num_problems, len(results[b]))):
            heat_data[p, i] = results[b][p]

    plt.imshow(heat_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='所需早停计数a')
    plt.xlabel('改进阈值b')
    plt.ylabel('问题ID')
    plt.title('不同问题和改进阈值b组合下所需的早停计数a')
    plt.xticks(range(len(b_values_sorted)), [f"{b:.6f}" for b in b_values_sorted], rotation=45)
    plt.yticks(range(num_problems), [f"问题{i + 1}" for i in range(num_problems)])
    plt.tight_layout()
    plt.savefig('heatmap_early_stopping.png')
    plt.close()


def find_best_parameters(results):
    """
    找出最佳参数组合

    参数:
    results - 分析结果字典

    返回:
    best_b, best_a - 最佳参数组合
    """
    # 计算每个b值下，所有问题所需的最小早停计数a
    min_a_for_all_problems = {b: max(counts) if counts else float('inf') for b, counts in results.items()}

    # 寻找最佳b值：使得a较小且对所有问题都适用
    # 计算每个b值的"性价比"：1/a
    efficiency = {b: 1 / a if a > 0 else 0 for b, a in min_a_for_all_problems.items()}
    best_b = max(efficiency, key=efficiency.get)
    best_a = min_a_for_all_problems[best_b]

    print(f"\n最佳参数组合: b = {best_b:.6f}, a = {best_a}")

    # 计算每个b值下的标准差，找出最稳定的参数
    std_devs = {b: np.std(counts) if counts else float('inf') for b, counts in results.items()}
    most_stable_b = min(std_devs, key=std_devs.get)
    print(f"最稳定的b值 (标准差最小): {most_stable_b:.6f}, 标准差 = {std_devs[most_stable_b]:.2f}")

    return best_b, best_a


if __name__ == "__main__":
    # 定义要测试的b值（改进阈值）
    b_values = [0.01, 0.05, 0.1]

    # 运行分析
    results = analyze_early_stopping(b_values)

    # 绘制结果
    plot_results(results, b_values)

    # 找出最佳参数
    best_b, best_a = find_best_parameters(results)

    # 输出早停实现示例
    print("\n早停机制实现示例:")
    print("""
def early_stopping(values, b_threshold={:.6f}, a_count={}):
    \"\"\"
    实现早停机制

    参数:
    values - 优化过程中的值序列
    b_threshold - 改进阈值，当相邻迭代改进小于此值时增加计数
    a_count - 早停计数阈值，连续计数达到此值时触发早停

    返回:
    bool - 是否应该早停
    \"\"\"
    consecutive_count = 0

    for i in range(1, len(values)):
        improvement = abs(values[i] - values[i-1])  # 计算相邻迭代的改进量

        if improvement <= b_threshold:
            consecutive_count += 1
            if consecutive_count >= a_count:
                return True
        else:
            consecutive_count = 0

    return False
    """.format(best_b, best_a))
