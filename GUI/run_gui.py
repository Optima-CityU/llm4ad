# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append('..')

import time
from datetime import datetime
import pytz
import tkinter as tk
from tkinter import ttk as tkttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
import webbrowser
import multiprocessing
from llm4ad.gui import main_gui
import threading
import ttkbootstrap as ttk
import subprocess
import yaml

##########################################################

selected_algo = None
selected_problem = None

process1 = None
thread1 = None

stop_thread = False
have_stop_thread = False

method_para_entry_list = []
batch_para_entry_list = []
method_para_value_type_list = []
method_para_value_name_list = []
batch_method_para = {} # 存了所有已选algo的参数
batch_problem_para = {} # 存了所有已选problem的参数

batch_last_para_type = 0
batch_last_para_name = None

problem_listbox = None
default_problem_index = None
objectives_var = None
problem_para_entry_list = []
problem_para_value_type_list = []
problem_para_value_name_list = []
problem_listbox2 = None # 左侧所有的task
objectives_var2 = None
algo_listbox2 = None # 左侧所有的algo
selected_algorithms_list = [] # 是个list，存了所有已经选了的algo的name
selected_tasks_list = [] # 是个list，存了所有已经选了的task的name
real_algo_listbox2 = None # 所有选择了的algo
real_prob_listbox2 = None # 所有选择了的prob
selected_problems_list = []

llm_para_entry_list = []
llm_para_entry_list2 = []
llm_para_value_name_list = ['name', 'host', 'key', 'model']
llm_para_default_value_list = ['HttpsApi', '', '', '']
llm_para_placeholder_list = ['HttpsApi', 'api.bltcy.top', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', 'gpt-4o-mini']

default_method = 'eoh'
default_problem = ['admissible_set', 'car_mountain', 'bactgrow']

tree = None

log_dir = None
figures = None
ax = None
canvas = None

batch_process1 = None
batch_thread1 = None
btach_log_dir = None

##########################################################

class PlaceholderEntry(ttk.Entry):
    def __init__(self, master=None, placeholder="Enter text here", color='grey', bootstyle='default', width=30):
        super().__init__(master, bootstyle=bootstyle, width=width)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['foreground']

        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)

        self._add_placeholder(force=True)

        self.have_content = False

    def _add_placeholder(self, event=None, force=False):
        self.have_content = True
        if not self.get() or force:
            self.configure(foreground=self.placeholder_color)
            self.delete(0, 'end')
            self.insert(0, self.placeholder)
            self.have_content = False

    def _clear_placeholder(self, event=None):
        if self.get() == self.placeholder and str(self['foreground']) == str(self.placeholder_color):
            self.delete(0, "end")
            self.configure(foreground=self.default_fg_color)


##########################################################

def draw_horizontal_line(parent_frame, width=150):
    line_canvas = tk.Canvas(parent_frame, width=width, height=25, bg='white', highlightthickness=0)
    line_canvas.pack(pady=0)
    line_canvas.create_line(0, 15, width, 15, fill='black')


def open_doc_link():
    webbrowser.open_new("https://llm4ad-doc.readthedocs.io/en/latest/")


def open_github_link():
    webbrowser.open_new("https://github.com/Optima-CityU/LLM4AD")


def open_website_link():
    webbrowser.open_new("http://www.llm4ad.com/index.html")


def open_qq_link():
    webbrowser.open_new("https://qm.qq.com/cgi-bin/qm/qr?k=4Imf8bn_d99-QXVcEJfOwCSD1KkcpbcD&jump_from=webapi&authKey=JtSmFh8BNKM97+TGnUdDgvT69TDTbo4UaLwgrZJSlsYqmVoCca/a5awU+TXt4zYB")


def open_folder():
    global log_dir

    if os.path.exists(log_dir):
        if os.name == 'nt':  # Windows
            os.startfile(log_dir)
        elif os.name == 'posix':  # Unix-like
            subprocess.run(['open', log_dir])


##########################################################

def add_algorithm():
    global algo_listbox2
    global selected_algorithms_list
    global real_algo_listbox2
    global batch_method_para

    selected = algo_listbox2.curselection()
    if not selected:
        return

    algo_name = algo_listbox2.get(selected[0])
    if algo_name not in selected_algorithms_list:
        real_algo_listbox2.insert(tk.END, algo_name)
        selected_algorithms_list.append(algo_name)
        required_parameters, value_type, default_value = get_required_parameters(
            path=f"../llm4ad/method/{algo_name}/paras.yaml")

        batch_method_para[algo_name] = {
            'name': required_parameters,
            'type': value_type,
            'value': [tk.StringVar(value=value) for value in default_value]
        }

def remove_algorithm():
    global selected_algorithms_list
    global real_algo_listbox2
    global batch_method_para
    global batch_last_para_type

    selected = real_algo_listbox2.curselection()
    if not selected:
        return

    batch_last_para_type = 0
    algo_name = real_algo_listbox2.get(selected[0])
    selected_algorithms_list.remove(algo_name)
    real_algo_listbox2.delete(selected[0])
    batch_method_para.pop(algo_name)

def add_problem():
    global problem_listbox2
    global selected_problems_list
    global real_prob_listbox2
    global batch_problem_para

    selected = problem_listbox2.curselection()
    if not selected:
        return

    prob_name = problem_listbox2.get(selected[0])
    prob_type = objectives_var2.get()
    prob_name = '{}/{}'.format(prob_type,prob_name)
    if prob_name not in selected_problems_list:
        real_prob_listbox2.insert(tk.END, prob_name)
        selected_problems_list.append(prob_name)
        required_parameters, value_type, default_value = get_required_parameters(
            path=f"../llm4ad/task/{prob_name}/paras.yaml")
        batch_problem_para[prob_name] = {
            'name': required_parameters,
            'type': value_type,
            'value': [tk.StringVar(value=value) for value in default_value]
        }

def remove_problem():
    global selected_problems_list
    global real_prob_listbox2
    global batch_problem_para
    global batch_last_para_type

    selected = real_prob_listbox2.curselection()
    if not selected:
        return

    batch_last_para_type = 0
    prob_name = real_prob_listbox2.get(selected[0])
    selected_problems_list.remove(prob_name)
    real_prob_listbox2.delete(selected[0])
    batch_problem_para.pop(prob_name)

############################

def batch_on_algo_select(event):
    if real_algo_listbox2.curselection():
        selected_algo = real_algo_listbox2.get(real_algo_listbox2.curselection())
        batch_show_algorithm_parameters(selected_algo)


def batch_on_problem_select(event):
    if real_prob_listbox2.curselection():
        selected_problem = real_prob_listbox2.get(real_prob_listbox2.curselection())
        batch_show_problem_parameters(selected_problem)


def batch_show_algorithm_parameters(algo_name):
    global batch_method_para
    global batch_para_entry_list
    global batch_last_para_type
    global batch_last_para_name
    batch_clear_param_frame()

    batch_last_para_type = 1
    batch_last_para_name = algo_name

    paras = batch_method_para[algo_name]
    required_parameters = paras['name']
    value_type = paras['type']
    default_value = paras['value']

    for i in range(len(required_parameters)):
        if i != 0:
            ttk.Label(para_setting_frame2, text=required_parameters[i] + ':').grid(row=i - 1, column=0, sticky='nsew', padx=5, pady=10)
        batch_para_entry_list.append(ttk.Entry(para_setting_frame2, width=10, bootstyle="primary"))
        if i != 0:
            batch_para_entry_list[-1].grid(row=i - 1, column=1, sticky='nsew', padx=5, pady=10)
            para_setting_frame2.grid_rowconfigure(i - 1, weight=1)
        if default_value[i] is not None:
            batch_para_entry_list[-1].insert(0, str(default_value[i].get()))
    para_setting_frame2.grid_columnconfigure(0, weight=1)
    para_setting_frame2.grid_columnconfigure(1, weight=2)

    if len(required_parameters) < 5:
        for i in range(len(required_parameters), 5):
            para_setting_frame2.grid_rowconfigure(i - 1, weight=1)


def batch_show_problem_parameters(problem_name):
    global batch_problem_para
    global batch_para_entry_list
    global batch_last_para_type
    global batch_last_para_name
    batch_clear_param_frame()

    batch_last_para_type = 2
    batch_last_para_name = problem_name

    paras = batch_problem_para[problem_name]
    required_parameters = paras['name']
    value_type = paras['type']
    default_value = paras['value']

    for i in range(len(required_parameters)):
        if i != 0:
            ttk.Label(para_setting_frame2, text=required_parameters[i] + ':').grid(row=i - 1, column=0, sticky='nsew',
                                                                                   padx=5, pady=10)
        batch_para_entry_list.append(ttk.Entry(para_setting_frame2, width=10, bootstyle="primary"))
        if i != 0:
            batch_para_entry_list[-1].grid(row=i - 1, column=1, sticky='nsew', padx=5, pady=10)
            para_setting_frame2.grid_rowconfigure(i - 1, weight=1)
        if default_value[i] is not None:
            batch_para_entry_list[-1].insert(0, str(default_value[i].get()))
    para_setting_frame2.grid_columnconfigure(0, weight=1)
    para_setting_frame2.grid_columnconfigure(1, weight=2)

    if len(required_parameters) < 5:
        for i in range(len(required_parameters), 5):
            para_setting_frame2.grid_rowconfigure(i - 1, weight=1)

def batch_clear_param_frame():
    global batch_para_entry_list
    global batch_last_para_type
    global batch_last_para_name
    global batch_problem_para
    global batch_method_para

    # 需要存上一次的东西
    if batch_last_para_type==0:
        pass
    elif batch_last_para_type==1:
        for i in range(len(batch_para_entry_list)):
            batch_method_para[batch_last_para_name]['value'][i].set(batch_para_entry_list[i].get())
    elif batch_last_para_type==2:
        for i in range(len(batch_para_entry_list)):
            batch_problem_para[batch_last_para_name]['value'][i].set(batch_para_entry_list[i].get())


    batch_para_entry_list = []
    for widget in para_setting_frame2.winfo_children():
        widget.destroy()


############################
def on_algo_select(event):
    global selected_algo
    if algo_listbox.curselection():
        selected_algo = algo_listbox.get(algo_listbox.curselection())
        show_algorithm_parameters(selected_algo)


def on_problem_select(event):
    global selected_problem
    if problem_listbox.curselection():
        selected_problem = problem_listbox.get(problem_listbox.curselection())
        show_problem_parameters(selected_problem)


def show_algorithm_parameters(algo_name):
    global method_para_entry_list
    global method_para_value_type_list
    global method_para_value_name_list
    clear_algo_param_frame()

    algo_param_frame['text'] = f"{algo_name}"

    required_parameters, value_type, default_value = get_required_parameters(path=f"../llm4ad/method/{algo_name}/paras.yaml")
    method_para_value_name_list = required_parameters
    method_para_value_type_list = value_type
    for i in range(len(required_parameters)):
        if i != 0:
            ttk.Label(algo_param_frame, text=required_parameters[i] + ':').grid(row=i - 1, column=0, sticky='nsew', padx=5, pady=10)
        method_para_entry_list.append(ttk.Entry(algo_param_frame, width=10, bootstyle="primary"))
        if i != 0:
            method_para_entry_list[-1].grid(row=i - 1, column=1, sticky='nsew', padx=5, pady=10)
            algo_param_frame.grid_rowconfigure(i - 1, weight=1)
        if default_value[i] is not None:
            method_para_entry_list[-1].insert(0, str(default_value[i]))
    algo_param_frame.grid_columnconfigure(0, weight=1)
    algo_param_frame.grid_columnconfigure(1, weight=2)

    if len(required_parameters) < 5:
        for i in range(len(required_parameters), 5):
            algo_param_frame.grid_rowconfigure(i - 1, weight=1)


def show_problem_parameters(problem_name):
    global problem_para_entry_list
    global problem_para_value_type_list
    global problem_para_value_name_list
    clear_problem_param_frame()

    problem_param_frame['text'] = f"{problem_name}"

    required_parameters, value_type, default_value = get_required_parameters(path=f"../llm4ad/task/{objectives_var.get()}/{problem_name}/paras.yaml")
    problem_para_value_type_list = value_type
    problem_para_value_name_list = required_parameters
    for i in range(len(required_parameters)):
        if i != 0:
            ttk.Label(problem_param_frame, text=required_parameters[i] + ':').grid(row=i - 1, column=0, sticky='nsew', padx=5, pady=10)
        problem_para_entry_list.append(ttk.Entry(problem_param_frame, width=10, bootstyle="warning"))
        if i != 0:
            problem_para_entry_list[-1].grid(row=i - 1, column=1, sticky='nsew', padx=5, pady=10)
            problem_param_frame.grid_rowconfigure(i - 1, weight=1)
        if default_value[i] is not None:
            problem_para_entry_list[-1].insert(0, str(default_value[i]))
    problem_param_frame.grid_columnconfigure(0, weight=1)
    problem_param_frame.grid_columnconfigure(1, weight=2)

    if len(required_parameters) < 5:
        for i in range(len(required_parameters), 5):
            problem_param_frame.grid_rowconfigure(i - 1, weight=1)


def get_required_parameters(path):
    required_parameters = []
    value_type = []
    default_value = []

    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)  # 使用 safe_load 读取 YAML 文件

    for key, value in data.items():
        required_parameters.append(key)
        value_type.append(str(type(value)))
        if value is None:
            default_value.append(value)
        else:
            default_value.append(str(value))

    return required_parameters, value_type, default_value


def clear_algo_param_frame():
    global method_para_entry_list
    global method_para_value_type_list
    global method_para_value_name_list
    method_para_value_type_list = []
    method_para_value_name_list = []
    method_para_entry_list = []
    for widget in algo_param_frame.winfo_children():
        widget.destroy()


def clear_problem_param_frame():
    global problem_para_entry_list
    global problem_para_value_type_list
    global problem_para_value_name_list
    problem_para_value_type_list = []
    problem_para_value_name_list = []
    problem_para_entry_list = []
    for widget in problem_param_frame.winfo_children():
        widget.destroy()


def problem_type_select(event=None):
    global problem_listbox
    global default_problem_index
    global objectives_var

    default_problem_index = None
    if problem_listbox is not None:
        problem_listbox.destroy()

    problem_listbox = tk.Listbox(problem_frame, height=6, bg='white', selectbackground='lightgray', font=('Comic Sans MS', 12))
    problem_listbox.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    path = f'../llm4ad/task/{objectives_var.get()}'
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and name != '__pycache__' and name != '_data':
            problem_listbox.insert(tk.END, name)
        if name in default_problem:
            default_problem_index = problem_listbox.size() - 1

    problem_listbox.bind("<<ListboxSelect>>", on_problem_select)
    on_problem_select(problem_listbox.select_set(default_problem_index))

def batch_exp_problem_type_select(event=None):
    global problem_listbox2
    global objectives_var2

    if problem_listbox2 is not None:
        problem_listbox2.destroy()

    problem_listbox2 = tk.Listbox(problem_frame2, height=6, bg='white', selectbackground='lightgray', font=('Comic Sans MS', 12))
    problem_listbox2.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    path = f'../llm4ad/task/{objectives_var2.get()}'
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and name != '__pycache__' and name != '_data':
            problem_listbox2.insert(tk.END, name)


###############################################################################

def batch_on_plot_button_click():
    # 启动后台线程执行任务队列
    task_thread = threading.Thread(target=batch_run)
    task_thread.start()

def batch_run():
    global batch_process1
    global batch_thread1
    global btach_log_dir

    try:
        if not batch_check_para():
            tk.messagebox.showinfo("Warning", "Please configure the settings of LLM.")
            return

        llm_para, method_para, problem_para, error = batch_return_para()

        if error:
            tk.messagebox.showinfo("Warning", "Please choose the algorithm and task you want to execute.")
            return

        init_table(list(method_para),list(problem_para))

        # todo 1
        #   功能
        #       stop按钮对应的方法
        #   存在的bug/异常处理
        #       如果运行中，进程出现bug?
        #       最重要的，不要有内存泄漏，以及停止了之后，直接把进程/线程kill掉，不要继续使用token了
        #       好像直接中止这个程序不会中止这个进程，这个进程还会继续。这个问题一定要解决，同时不知道single(非batch)的会不会有这个问题
        #       如果卡死，则需要加队列(queue)，具体看deepseek
        #       如果加入了algo/task，然后删掉加入的，使得algo/task为空，也是能跑的。
        #       点击run按钮，会保存最后一次修改的参数，有没有什么情况，会导致报错（比如从来没点击过添加的algo/task,加入进来直接run了)，会不会报错？等等
        #   功能
        #       结果显示，表格最好能点开看具体的运行配置和结果（log_files)

        plot_button2['state'] = tk.DISABLED
        stop_button2['state'] = tk.NORMAL
        for row_index, (method_key, method_value) in enumerate(method_para.items()):
            for col_index, (problem_key, problem_value) in enumerate(problem_para.items()):
                profiler_para = {}
                profiler_para['name'] = 'ProfilerBase'
                temp_str1 = problem_value['name']
                temp_str2 = method_value['name']
                process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
                b = os.path.abspath('..')
                log_folder = b + '/GUI/logs/' + process_start_time.strftime(
                    "%Y%m%d_%H%M%S") + f'_{temp_str1}' + f'_{temp_str2}'
                profiler_para['log_dir'] = log_folder

                #########

                process = multiprocessing.Process(target=main_gui, args=(llm_para, method_value, problem_value, profiler_para))
                # thread = threading.Thread(target=get_results, args=(profiler_para['log_dir'], method_para['max_sample_nums'],))
                thread = threading.Thread(target=batch_get_results,args=(profiler_para['log_dir'],method_value['max_sample_nums'],row_index,col_index))

                process.start()
                thread.start()

                process.join()
                thread.join()
        plot_button2['state'] = tk.NORMAL
        stop_button2['state'] = tk.DISABLED


    except ValueError:
        print("Invalid input.")

def init_table(methods_name, problems_name):
    global tree

    for widget in right_frame2.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(right_frame2)
    tree["columns"] = problems_name
    # 配置行表头列（第一列）
    tree.column("#0", width=100, minwidth=100, anchor=tk.CENTER)
    tree.heading("#0", text="", anchor=tk.CENTER)
    # 配置其他列（列表头）
    for col in problems_name:
        tree.column(col, width=100, minwidth=100, anchor=tk.CENTER)
        tree.heading(col, text=col, anchor=tk.CENTER)
    # 添加行数据（包括行表头）
    for i, row_header in enumerate(methods_name):
        # 使用行表头作为第一列的值，其余列为空
        values = [""] * len(problems_name)
        tree.insert("", i, text=row_header, values=values)
    tree.pack(padx=10, pady=10)

def batch_return_para():
    global batch_method_para
    global batch_problem_para

    llm_para = {}
    method_para = {}
    problem_para = {}
    error = False

    batch_clear_param_frame()

    ####################

    if len(batch_method_para) == 0:
        error = True
        return llm_para, method_para, problem_para, error
    if len(batch_problem_para) == 0:
        error = True
        return llm_para, method_para, problem_para, error

    for i in range(len(llm_para_entry_list2)):
        llm_para[llm_para_value_name_list[i]] = llm_para_entry_list2[i].get()

    for key, value in batch_method_para.items():
        method_para[key] = {}
        for i in range(len(value['name'])):
            method_para[key][value['name'][i]] = value['value'][i].get()
            if value['type'][i] == '<class \'int\'>':
                method_para[key][value['name'][i]] = int(value['value'][i].get())

        method_para[key]['num_samplers'] = method_para[key]['num_evaluators']

    for key, value in batch_problem_para.items():
        problem_para[key] = {}
        for i in range(len(value['name'])):
            problem_para[key][value['name'][i]] = value['value'][i].get()
            if value['type'][i] == '<class \'int\'>':
                problem_para[key][value['name'][i]] = int(value['value'][i].get())

    ####################

    print(llm_para)
    print(method_para)
    print(problem_para)

    return llm_para, method_para, problem_para, error

def batch_check_para():
    for i in llm_para_entry_list2[1:]:
        if not i.have_content:
            return False
    return True

def batch_get_results(log_dir, max_sample_nums,row_index,col_index):
    index = 1

    while (not check_finish(log_dir, index, max_sample_nums)):
        time.sleep(0.5)
        new = check(index, log_dir)
        if new:
            new_value = batch_get_latest_result(index, log_dir)
            update_cell_value(row_index, col_index, new_value)

            index += 1

def batch_get_latest_result(index, log_dir):
    generation = []
    best_value_list = []
    all_best_value = float('-inf')
    best_alg = None

    file_name_list = [log_dir + f'/samples/samples_{i * 200 + 1}~{(i + 1) * 200}.json' for i in
                      range(((index - 1) // 200) + 1)]

    data = []
    for file_name in file_name_list:
        with open(file_name) as file:
            data.append(json.load(file))

    for i in range(index):
        individual = data[i // 200][((i + 1) % 200) - 1]
        code = individual['function']
        # alg = individual['algorithm']
        obj = individual['score']
        if obj is None:
            generation.append(i + 1)
            best_value_list.append(all_best_value)
            continue
        if obj > all_best_value:
            all_best_value = obj
            best_alg = code
        generation.append(i + 1)
        best_value_list.append(all_best_value)

    generation = np.array(generation)
    best_value_list = np.array(best_value_list)

    return all_best_value

def update_cell_value(row_index, col_index, new_value):
    global tree
    row_id = tree.get_children()[row_index]
    current_values = list(tree.item(row_id, 'values'))
    current_values[col_index] = new_value
    tree.item(row_id, values=current_values)


######################################################################

def on_plot_button_click():
    global process1
    global thread1
    global log_dir

    try:

        if not check_para():
            tk.messagebox.showinfo("Warning", "Please configure the settings of LLM.")
            return

        llm_para, method_para, problem_para, profiler_para = return_para()

        init_fig(method_para['max_sample_nums'])

        process1 = multiprocessing.Process(target=main_gui, args=(llm_para, method_para, problem_para, profiler_para))
        process1.start()

        thread1 = threading.Thread(target=get_results, args=(profiler_para['log_dir'], method_para['max_sample_nums'],), daemon=True)
        thread1.start()

        log_dir = profiler_para['log_dir']

        plot_button['state'] = tk.DISABLED
        stop_button['state'] = tk.NORMAL
        # doc_button['state'] = tk.DISABLED
        doc_button['state'] = tk.NORMAL

    except ValueError:
        print("Invalid input.")

def check_para():
    for i in llm_para_entry_list[1:]:
        if not i.have_content:
            return False
    return True

def return_para():
    llm_para = {}
    method_para = {}
    problem_para = {}
    profiler_para = {}

    ####################

    for i in range(len(llm_para_entry_list)):
        llm_para[llm_para_value_name_list[i]] = llm_para_entry_list[i].get()

    for i in range(len(method_para_entry_list)):
        method_para[method_para_value_name_list[i]] = method_para_entry_list[i].get()
        if method_para_value_type_list[i] == '<class \'int\'>':
            method_para[method_para_value_name_list[i]] = int(method_para_entry_list[i].get())

    method_para['num_samplers'] = method_para['num_evaluators']

    for i in range(len(problem_para_entry_list)):
        problem_para[problem_para_value_name_list[i]] = problem_para_entry_list[i].get()
        if problem_para_value_type_list[i] == '<class \'int\'>':
            problem_para[problem_para_value_name_list[i]] = int(problem_para_entry_list[i].get())

    ####################

    profiler_para['name'] = 'ProfilerBase'

    temp_str1 = problem_para['name']
    temp_str2 = method_para['name']
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    b = os.path.abspath('..')
    log_folder = b + '/GUI/logs/' + process_start_time.strftime(
        "%Y%m%d_%H%M%S") + f'_{temp_str1}' + f'_{temp_str2}'
    profiler_para['log_dir'] = log_folder

    ####################

    print(llm_para)
    print(method_para)
    print(problem_para)
    print(profiler_para)

    return llm_para, method_para, problem_para, profiler_para

def init_fig(max_sample_nums):
    global stop_thread
    global have_stop_thread
    global thread1
    global process1
    global ax
    global figures
    global canvas

    stop_run()
    value_label.config(text=f"{0} samples")

    stop_thread = False
    have_stop_thread = False

    right_frame_label['text'] = 'Running'

    code_display.config(state='normal')
    code_display.delete(1.0, 'end')
    code_display.config(state='disabled')

    objective_label['text'] = 'Current best objective:'

    for widget in plot_frame.winfo_children():
        widget.destroy()

    font = {
        'family': 'Times New Roman',
        'size': 16
    }

    figures = plt.Figure(figsize=(4, 3), dpi=100)
    ax = figures.add_subplot(111)

    figures.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"Result Display", fontdict=font)

    ax.plot()
    ax.set_xlim(left=0)
    ax.set_xlabel('Samples', fontdict=font)
    ax.set_ylabel('Current best objective', fontdict=font)
    ax.grid(True)

    if max_sample_nums <= 20:
        ax.set_xticks(np.arange(0, max_sample_nums + 1, 1))
    else:
        ticks = np.linspace(0, max_sample_nums, 11)
        ticks = np.round(ticks).astype(int)
        ax.set_xticks(ticks)

    canvas = FigureCanvasTkAgg(figures, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

######################################################################

def get_results(log_dir, max_sample_nums):
    global figures
    global stop_thread
    global have_stop_thread
    index = 1

    while (not stop_thread) and (not check_finish(log_dir, index, max_sample_nums)) and (not except_error()):
        time.sleep(0.5)
        new = check(index, log_dir)
        if new:
            try:
                fig, alg, best_obj = plot_fig(index, log_dir, max_sample_nums)
            except:
                continue
            display_plot(index - 1)
            if alg is not None:
                display_alg(alg)
            objective_label['text'] = f'Current best objective:{best_obj}'
            index += 1

    if not stop_thread:
        right_frame_label['text'] = 'Finished'
        # doc_button['state'] = tk.NORMAL

    if except_error():
        tk.messagebox.showerror("Error", "Except Error. Please check the terminal.")
        right_frame_label['text'] = 'Error'

    have_stop_thread = True
    plot_button['state'] = tk.NORMAL
    stop_button['state'] = tk.DISABLED

def plot_fig(index, log_dir, max_sample_nums):
    global figures
    global ax
    ###############################################################
    generation = []
    best_value_list = []
    all_best_value = float('-inf')
    best_alg = None

    file_name_list = [log_dir + f'/samples/samples_{i * 200 + 1}~{(i + 1) * 200}.json' for i in range(((index - 1) // 200) + 1)]

    data = []
    for file_name in file_name_list:
        with open(file_name) as file:
            data.append(json.load(file))

    for i in range(index):
        individual = data[i // 200][((i+1) % 200)-1]
        code = individual['function']
        # alg = individual['algorithm']
        obj = individual['score']
        if obj is None:
            generation.append(i + 1)
            best_value_list.append(all_best_value)
            continue
        if obj > all_best_value:
            all_best_value = obj
            best_alg = code
        generation.append(i + 1)
        best_value_list.append(all_best_value)

    generation = np.array(generation)
    best_value_list = np.array(best_value_list)

    ###############################################################
    # plot

    font = {
        'family': 'Times New Roman',
        'size': 16
    }

    figures.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"Result display", fontdict=font)

    # ax.plot(generation, best_value_list, color='tab:blue', marker='o')
    ax.plot(generation, best_value_list, color='tab:blue')
    ax.set_xlabel('Samples', fontdict=font)
    ax.set_ylabel('Current best objective', fontdict=font)
    ax.grid(True)

    if len(generation) <= max_sample_nums:
        if max_sample_nums<=20:
            ax.set_xticks(np.arange(0, max_sample_nums + 1, 1))
        else:
            ticks = np.linspace(0, max_sample_nums, 11)
            ticks = np.round(ticks).astype(int)
            ax.set_xticks(ticks)
    else:
        if len(generation)<=20:
            ax.set_xticks(np.arange(0, len(generation) + 1, 1))
        else:
            ticks = np.linspace(0, len(generation), 11)
            ticks = np.round(ticks).astype(int)
            ax.set_xticks(ticks)

    ###############################################################

    return figures, best_alg, all_best_value

def display_plot(index):
    global canvas
    canvas.draw()

    value_label.config(text=f"{index + 1} samples")

def display_alg(alg):
    code_display.config(state='normal')
    code_display.delete(1.0, 'end')
    code_display.insert(tk.END, alg)
    code_display.config(state='disabled')

def except_error():
    global process1
    try:
        if process1.exitcode == 1:
            return True
        else:
            return False
    except:
        return False

def check_finish(log_dir, index, max_sample_nums):
    return os.path.exists(log_dir + '/population/' + 'end.json') or index > max_sample_nums

def check(index, log_dir):
    temp_var1 = (index - 1) // 200
    return_value = False
    file_name = log_dir + f'/samples/samples_{temp_var1*200+1}~{(temp_var1+1)*200}.json'

    if os.path.exists(file_name):
        with open(file_name) as file:
            data = json.load(file)
        if len(data) >= ((index-1) % 200)+1:
            return_value = True
    return return_value

######################################################################

def stop_run_thread():
    thread_stop = threading.Thread(target=stop_run)
    thread_stop.start()

def stop_run():
    global stop_thread
    global process1
    global have_stop_thread

    # doc_button['state'] = tk.DISABLED
    stop_button['state'] = tk.DISABLED
    stop_thread = True
    if process1 is not None:
        if process1.is_alive():
            try:
                process1.terminate()
            except:
                pass
    while (thread1 is not None) and (have_stop_thread is False):
        time.sleep(0.5)
        _ = 'stop'
    plot_button['state'] = tk.NORMAL

def exit_run():
    stop_run_thread()
    root.destroy()
    sys.exit(0)

######################################################################

def show_frame(frame, button):
    frame.tkraise()

###############################################################################

if __name__ == '__main__':

    root = ttk.Window()
    root.title("LLM4AD")
    root.geometry("1500x900")
    root.protocol("WM_DELETE_WINDOW", exit_run)

    root.iconbitmap('./image/icon.ico')

    style = tkttk.Style()
    style.configure("TLabelframe.Label", font=('Helvetica', 15))
    style.configure("TLabel", font=('Comic Sans MS', 12))
    style.configure("TCombobox", font=('Comic Sans MS', 10))

    photo_doc = tk.PhotoImage(file=r"./image/document.png")
    photoimage_doc = photo_doc.subsample(10, 10)
    photo_web = tk.PhotoImage(file=r"./image/website.png")
    photoimage_web = photo_web.subsample(10, 10)
    photo_git = tk.PhotoImage(file=r"./image/github.png")
    photoimage_git = photo_git.subsample(10, 10)
    photo_qq = tk.PhotoImage(file=r"./image/qq.png")
    photoimage_qq = photo_qq.subsample(10, 10)

    top_frame = ttk.Frame(root, height=30, bootstyle="info")
    top_frame.pack(fill='x')

    # 创建底部 Frame
    bottom_frame = ttk.Frame(root)
    bottom_frame.pack(fill='both', expand=True)

    # 创建两个 Frame 用于切换
    frame1 = ttk.Frame(bottom_frame)
    frame2 = ttk.Frame(bottom_frame)

    for frame in (frame1, frame2):
        frame.grid(row=0, column=0, sticky='nsew')

    bottom_frame.grid_rowconfigure(0, weight=1)
    bottom_frame.grid_columnconfigure(0, weight=1)

    button1 = ttk.Button(top_frame, text="Single Experiment", command=lambda: show_frame(frame1, button1))
    button1.pack(side=tk.LEFT, padx=3)

    button2 = ttk.Button(top_frame, text="Batch Experiments", command=lambda: show_frame(frame2, button2))
    button2.pack(side=tk.LEFT, padx=3)

    link_doc = ttk.Button(top_frame, image=photoimage_doc, command=open_doc_link, bootstyle="info")
    link_doc.pack(side=ttk.RIGHT, padx=3)
    link_git = ttk.Button(top_frame, image=photoimage_git, command=open_github_link, bootstyle="info")
    link_git.pack(side=ttk.RIGHT, padx=3)
    link_web = ttk.Button(top_frame, image=photoimage_web, command=open_website_link, bootstyle="info")
    link_web.pack(side=ttk.RIGHT, padx=3)
    link_qq = ttk.Button(top_frame, image=photoimage_qq, command=open_qq_link, bootstyle="info")
    link_qq.pack(side=ttk.RIGHT, padx=3)

    ##########################################################

    left_frame = ttk.Frame(frame1)
    left_frame.grid(row=0, column=0, sticky="nsew")
    ttk.Separator(frame1, orient='vertical', bootstyle="secondary").grid(row=0, column=1, sticky="ns")
    right_frame = ttk.Frame(frame1)
    right_frame.grid(row=0, column=2, sticky="nsew")

    frame1.grid_rowconfigure(0, weight=1)
    frame1.grid_columnconfigure(0, weight=2)
    frame1.grid_columnconfigure(1, weight=1)
    frame1.grid_columnconfigure(2, weight=30)


    #####################################################

    llm_frame = ttk.Labelframe(left_frame, text="LLM setups", bootstyle="dark")
    llm_frame.pack(anchor=tk.NW, fill=tk.X, padx=5, pady=5)

    for i in range(len(llm_para_value_name_list)):
        llm_para_entry_list.append(PlaceholderEntry(llm_frame, width=70, bootstyle="dark", placeholder=llm_para_placeholder_list[i]))
        if i != 0:
            ttk.Label(llm_frame, text=llm_para_value_name_list[i] + ':').grid(row=i - 1, column=0, sticky='ns', padx=5, pady=5)
            llm_para_entry_list[-1].grid(row=i - 1, column=1, sticky='ns', padx=5, pady=5)
            llm_frame.grid_rowconfigure(i - 1, weight=1)

    llm_frame.grid_columnconfigure(0, weight=1)
    llm_frame.grid_columnconfigure(1, weight=1)

    with_default_parameter = False
    if with_default_parameter:
        for i in range(len(llm_para_value_name_list)):
            llm_para_entry_list[i].delete(0, 'end')
            llm_para_entry_list[i].configure(foreground=llm_para_entry_list[i].default_fg_color)
            llm_para_entry_list[i].insert(0, str(llm_para_default_value_list[i]))
    else:
        llm_para_entry_list[0].delete(0, 'end')
        llm_para_entry_list[0].configure(foreground=llm_para_entry_list[0].default_fg_color)
        llm_para_entry_list[0].insert(0, str(llm_para_default_value_list[0]))

    ############

    container_frame_1 = tk.Frame(left_frame)
    container_frame_1.pack(fill=tk.BOTH, expand=True)

    algo_frame = ttk.Labelframe(container_frame_1, text="Methods", bootstyle="primary")
    algo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    problem_frame = ttk.Labelframe(container_frame_1, text="Tasks", bootstyle="warning")
    problem_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    ############

    container_frame_2 = tk.Frame(left_frame)
    container_frame_2.pack(fill=tk.BOTH, expand=True)

    algo_param_frame = ttk.Labelframe(container_frame_2, text="eoh", bootstyle="primary")
    algo_param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    problem_param_frame = ttk.Labelframe(container_frame_2, text="admissible_set", bootstyle="warning")
    problem_param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    ############

    algo_listbox = tk.Listbox(algo_frame, height=6, bg='white', selectbackground='lightgray', font=('Comic Sans MS', 12))
    algo_listbox.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    default_method_index = None
    path = '../llm4ad/method'
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and name != '__pycache__':
            algo_listbox.insert(tk.END, name)
        if name == default_method:
            default_method_index = algo_listbox.size() - 1

    algo_listbox.bind("<<ListboxSelect>>", on_algo_select)
    on_algo_select(algo_listbox.select_set(default_method_index))

    ############

    objectives_var = tk.StringVar(value="optimization")
    objectives_frame = tk.Frame(problem_frame, bg='white')
    objectives_frame.pack(anchor=tk.NW, pady=5)
    radiobutton_list = []
    for _, dict_name, _ in os.walk('../llm4ad/task'):
        for name in dict_name:
            if name != '__pycache__' and name != '_data':
                radiobutton_list.append(name)
        break
    combobox = ttk.Combobox(objectives_frame, state='readonly', values=radiobutton_list, textvariable=objectives_var, bootstyle="warning", font=('Comic Sans MS', 12))
    combobox.bind('<<ComboboxSelected>>', problem_type_select)
    combobox.pack(anchor=tk.NW, padx=5, pady=5)
    problem_type_select()

    ############

    plot_button = ttk.Button(left_frame, text="Run", command=on_plot_button_click, width=12, bootstyle="primary-outline", state=tk.NORMAL)
    plot_button.pack(side='left', pady=20, expand=True)

    stop_button = ttk.Button(left_frame, text="Stop", command=stop_run_thread, width=12, bootstyle="warning-outline", state=tk.DISABLED)
    stop_button.pack(side='left', pady=20, expand=True)

    doc_button = ttk.Button(left_frame, text="Log files", command=open_folder, width=12, bootstyle="dark-outline", state=tk.DISABLED)
    doc_button.pack(side='left', pady=20, expand=True)

    ##########################################################

    state_frame = ttk.Frame(right_frame)
    state_frame.grid(row=0, column=0, sticky='ns', padx=5, pady=5)
    code_frame = ttk.Frame(right_frame)
    code_frame.grid(row=0, column=1, sticky='ns', padx=5, pady=5)

    plot_frame = tk.Frame(right_frame, bg='white')
    plot_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

    right_frame.grid_rowconfigure(0, weight=400)
    right_frame.grid_rowconfigure(1, weight=2500)
    right_frame.grid_columnconfigure(0, weight=400)
    right_frame.grid_columnconfigure(1, weight=600)

    ###

    right_frame_label = ttk.Label(state_frame, text="Wait", anchor='w')
    right_frame_label.pack(fill=tk.X, padx=10, pady=10)

    value_label = ttk.Label(state_frame, text="0 samples", anchor='w')
    value_label.pack(fill=tk.X, padx=10, pady=10)

    objective_label = ttk.Label(state_frame, text="Current best objective:", anchor='w')
    objective_label.pack(fill=tk.X, padx=10, pady=10)

    ###

    code_display_frame = ttk.Labelframe(code_frame, text="Current best algorithm:", bootstyle="dark")
    code_display_frame.pack(anchor=tk.NW, fill=tk.X, padx=5, pady=5)
    code_display = tk.Text(code_display_frame, height=14, width=70)
    code_display.pack(fill='both', expand=True, padx=5, pady=5)
    sorting_algorithm = ""
    code_display.insert(tk.END, sorting_algorithm)
    code_display.config(state='disabled')

    ##########################################################

    # frame 2

    left_frame2 = ttk.Frame(frame2)
    left_frame2.grid(row=0, column=0, sticky="nsew")
    ttk.Separator(frame2, orient='vertical', bootstyle="secondary").grid(row=0, column=1, sticky="ns")
    right_frame2 = ttk.Frame(frame2)
    right_frame2.grid(row=0, column=2, sticky="nsew")

    frame2.grid_rowconfigure(0, weight=1)
    frame2.grid_columnconfigure(0, weight=2)
    frame2.grid_columnconfigure(1, weight=1)
    frame2.grid_columnconfigure(2, weight=30)

    ###

    llm_frame2 = ttk.Labelframe(left_frame2, text="LLM setups", bootstyle="dark")
    llm_frame2.pack(anchor=tk.NW, fill=tk.X, padx=5, pady=5)

    for i in range(len(llm_para_value_name_list)):
        llm_para_entry_list2.append(
            PlaceholderEntry(llm_frame2, width=70, bootstyle="dark", placeholder=llm_para_placeholder_list[i]))
        if i != 0:
            ttk.Label(llm_frame2, text=llm_para_value_name_list[i] + ':').grid(row=i - 1, column=0, sticky='ns', padx=5,
                                                                              pady=5)
            llm_para_entry_list2[-1].grid(row=i - 1, column=1, sticky='ns', padx=5, pady=5)
            llm_frame2.grid_rowconfigure(i - 1, weight=1)

    llm_frame2.grid_columnconfigure(0, weight=1)
    llm_frame2.grid_columnconfigure(1, weight=1)

    with_default_parameter = False
    if with_default_parameter:
        for i in range(len(llm_para_value_name_list)):
            llm_para_entry_list2[i].delete(0, 'end')
            llm_para_entry_list2[i].configure(foreground=llm_para_entry_list2[i].default_fg_color)
            llm_para_entry_list2[i].insert(0, str(llm_para_default_value_list[i]))
    else:
        llm_para_entry_list2[0].delete(0, 'end')
        llm_para_entry_list2[0].configure(foreground=llm_para_entry_list2[0].default_fg_color)
        llm_para_entry_list2[0].insert(0, str(llm_para_default_value_list[0]))

    ###

    container_frame_1_2 = tk.Frame(left_frame2)
    container_frame_1_2.pack(fill=tk.BOTH, expand=True)

    algo_frame2 = ttk.Labelframe(container_frame_1_2, text="Methods", bootstyle="primary")
    algo_frame2.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    button_frame_algo_2 = tk.Frame(container_frame_1_2)
    button_frame_algo_2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    add_button2 = ttk.Button(button_frame_algo_2, text="Add-->", width=12, command=add_algorithm,
                              bootstyle="primary-outline", state=tk.NORMAL)
    add_button2.grid(row=1, column=1, pady=5)
    delete_button2 = ttk.Button(button_frame_algo_2, text="<--Delete", width=12, command=remove_algorithm,
                             bootstyle="primary-outline", state=tk.NORMAL)
    delete_button2.grid(row=3, column=1, pady=5)

    button_frame_algo_2.grid_rowconfigure(0, weight=6)
    button_frame_algo_2.grid_rowconfigure(1, weight=10)
    button_frame_algo_2.grid_rowconfigure(2, weight=6)
    button_frame_algo_2.grid_rowconfigure(3, weight=10)
    button_frame_algo_2.grid_rowconfigure(4, weight=6)
    button_frame_algo_2.grid_columnconfigure(0, weight=2)
    button_frame_algo_2.grid_columnconfigure(1, weight=4)
    button_frame_algo_2.grid_columnconfigure(2, weight=2)

    real_algo_frame2 = ttk.Labelframe(container_frame_1_2, text="Selected Methods", bootstyle="warning")
    real_algo_frame2.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

    container_frame_1_2.grid_rowconfigure(0, weight=10)
    container_frame_1_2.grid_columnconfigure(0, weight=10)
    container_frame_1_2.grid_columnconfigure(1, weight=3)
    container_frame_1_2.grid_columnconfigure(2, weight=10)

    real_algo_listbox2 = tk.Listbox(real_algo_frame2, height=6, bg='white', selectbackground='lightgray',
                               font=('Comic Sans MS', 12))
    real_algo_listbox2.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    real_algo_listbox2.bind("<<ListboxSelect>>", batch_on_algo_select)

    ###

    algo_listbox2 = tk.Listbox(algo_frame2, height=6, bg='white', selectbackground='lightgray',
                              font=('Comic Sans MS', 12))
    algo_listbox2.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    for _, dict_name, _ in os.walk('../llm4ad/method'):
        for name in dict_name:
            if name != '__pycache__':
                algo_listbox2.insert(tk.END, name)

    ###

    container_frame_2_2 = tk.Frame(left_frame2)
    container_frame_2_2.pack(fill=tk.BOTH, expand=True)

    problem_frame2 = ttk.Labelframe(container_frame_2_2, text="Tasks", bootstyle="primary")
    problem_frame2.grid(row=0, column=0, sticky="nsew")

    button_frame_prob_2 = tk.Frame(container_frame_2_2)
    button_frame_prob_2.grid(row=0, column=1, sticky="nsew")

    add_button22 = ttk.Button(button_frame_prob_2, text="Add-->", width=12, command=add_problem,
                             bootstyle="primary-outline", state=tk.NORMAL)
    add_button22.grid(row=1, column=1, pady=5)
    delete_button22 = ttk.Button(button_frame_prob_2, text="<--Delete", width=12, command=remove_problem,
                                bootstyle="primary-outline", state=tk.NORMAL)
    delete_button22.grid(row=3, column=1, pady=5)

    button_frame_prob_2.grid_rowconfigure(0, weight=6)
    button_frame_prob_2.grid_rowconfigure(1, weight=10)
    button_frame_prob_2.grid_rowconfigure(2, weight=6)
    button_frame_prob_2.grid_rowconfigure(3, weight=10)
    button_frame_prob_2.grid_rowconfigure(4, weight=6)
    button_frame_prob_2.grid_columnconfigure(0, weight=2)
    button_frame_prob_2.grid_columnconfigure(1, weight=4)
    button_frame_prob_2.grid_columnconfigure(2, weight=2)

    real_problem_frame2 = ttk.Labelframe(container_frame_2_2, text="Selected Tasks", bootstyle="warning")
    real_problem_frame2.grid(row=0, column=2, sticky="nsew")

    container_frame_2_2.grid_rowconfigure(0, weight=10)
    container_frame_2_2.grid_columnconfigure(0, weight=10)
    container_frame_2_2.grid_columnconfigure(1, weight=3)
    container_frame_2_2.grid_columnconfigure(2, weight=10)

    real_prob_listbox2 = tk.Listbox(real_problem_frame2, height=6, bg='white', selectbackground='lightgray',
                                    font=('Comic Sans MS', 12))
    real_prob_listbox2.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    real_prob_listbox2.bind("<<ListboxSelect>>", batch_on_problem_select)


    ###

    objectives_var2 = tk.StringVar(value="optimization")
    objectives_frame2 = tk.Frame(problem_frame2, bg='white')
    objectives_frame2.pack(anchor=tk.NW, pady=5)
    radiobutton_list2 = []
    for _, dict_name, _ in os.walk('../llm4ad/task'):
        for name in dict_name:
            if name != '__pycache__' and name != '_data':
                radiobutton_list2.append(name)
        break
    combobox2 = ttk.Combobox(objectives_frame2, state='readonly', values=radiobutton_list2, textvariable=objectives_var2,
                            bootstyle="warning", font=('Comic Sans MS', 12))
    combobox2.pack(anchor=tk.NW, padx=5, pady=5)

    combobox2.bind('<<ComboboxSelected>>', batch_exp_problem_type_select)
    batch_exp_problem_type_select()

    ###

    container_frame_3_2 = tk.Frame(left_frame2)
    container_frame_3_2.pack(fill=tk.BOTH, expand=True)

    para_setting_frame2 = ttk.Labelframe(container_frame_3_2, text="Parameter Settings", bootstyle="primary")
    para_setting_frame2.grid(row=0, column=0, sticky="nsew")

    container_frame_3_2.grid_rowconfigure(0, weight=10)
    container_frame_3_2.grid_columnconfigure(0, weight=10)

    ##########################################################

    # todo 1 做到这里了（为按钮绑定函数），上面的组件都不用管

    # plot_button2 = ttk.Button(left_frame2, text="Run", command=on_plot_button_click, width=12,
    #                          bootstyle="primary-outline", state=tk.NORMAL)
    plot_button2 = ttk.Button(left_frame2, text="Run", width=12, command=batch_on_plot_button_click,
                              bootstyle="primary-outline", state=tk.NORMAL)
    plot_button2.pack(side='left', pady=20, expand=True)

    # stop_button2 = ttk.Button(left_frame2, text="Stop", command=stop_run_thread, width=12, bootstyle="warning-outline",
    #                          state=tk.DISABLED)
    stop_button2 = ttk.Button(left_frame2, text="Stop", width=12, bootstyle="warning-outline",
                              state=tk.DISABLED)
    stop_button2.pack(side='left', pady=20, expand=True)

    ##########################################################

    # 默认显示 Frame 1，并设置按钮状态
    show_frame(frame1, button1)

    ##########################################################

    root.mainloop()

    ##########################################################