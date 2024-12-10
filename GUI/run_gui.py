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

##########################################################

selected_algo = None
selected_problem = None

process1 = None
thread1 = None

stop_thread = False
have_stop_thread = False

method_para_entry_list = []
method_para_value_type_list = []
method_para_value_name_list = []

problem_listbox = None
default_problem_index = None
objectives_var = None
problem_para_entry_list = []
problem_para_value_type_list = []
problem_para_value_name_list = []

llm_para_entry_list = []
llm_para_value_name_list = ['name', 'host', 'key', 'model']
llm_para_default_value_list = ['HttpsApi', '', '', '']
llm_para_placeholder_list = ['HttpsApi', 'api.bltcy.top', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', 'gpt-4o-mini-2024-07-18']

default_method = 'eoh'
default_problem = ['online_bin_packing', 'car_mountain', 'bactgrow']

log_dir = None
figures = []


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

    required_parameters, value_type, default_value = get_required_parameters(path=f"../llm4ad/method/{algo_name}/{algo_name}.py")
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

    required_parameters, value_type, default_value = get_required_parameters(path=f"../llm4ad/task/{objectives_var.get()}/{problem_name}/evaluation.py")
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

    for line in open(path, "r",encoding="utf-8").readlines():
        if line == '# end\n' or line[0] != '#':
            break
        if line[:13] == '# Parameters:':
            continue
        line = line[:-1]
        line = line[2:]
        line = line.split(": ")
        required_parameters.append(line[0])
        value_type.append(line[1])
        if line[2] == '':
            default_value.append(None)
        else:
            default_value.append(line[2])

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
    for _, dict_name, _ in os.walk(f'../llm4ad/task/{objectives_var.get()}'):
        for name in dict_name:
            if name != '__pycache__' and name != '_data':
                problem_listbox.insert(tk.END, name)
            if name in default_problem:
                default_problem_index = problem_listbox.size() - 1
    problem_listbox.bind("<<ListboxSelect>>", on_problem_select)
    on_problem_select(problem_listbox.select_set(default_problem_index))


###############################################################################

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
        print("Invalid input. Please enter a number.")


def check_para():
    for i in llm_para_entry_list[1:]:
        if not i.have_content:
            return False
    return True


def init_fig(max_sample_nums):
    global stop_thread
    global have_stop_thread
    global thread1
    global process1

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

    fig = plt.Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"Result Display", fontdict=font)

    ax.plot()
    ax.set_xlim(left=0)
    ax.set_xlabel('Samples', fontdict=font)
    ax.set_ylabel('Current best objective', fontdict=font)
    ax.grid(True)

    ax.set_xticks(np.arange(0, max_sample_nums + 1, 1))

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


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
        if method_para_value_type_list[i] == 'int':
            method_para[method_para_value_name_list[i]] = int(method_para_entry_list[i].get())

    method_para['num_samplers'] = method_para['num_evaluators']

    for i in range(len(problem_para_entry_list)):
        problem_para[problem_para_value_name_list[i]] = problem_para_entry_list[i].get()
        if problem_para_value_type_list[i] == 'int':
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


def get_results(log_dir, max_sample_nums):
    global figures
    global stop_thread
    global have_stop_thread
    figures = []
    index = 1

    while (not stop_thread) and (not check_finish(log_dir, index, max_sample_nums)) and (not except_error()):
        time.sleep(0.5)
        new = check(index, log_dir)
        if new:
            try:
                fig, alg, best_obj = plot_fig(index, log_dir, max_sample_nums)
            except:
                continue
            figures.append(fig)
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
    return_value = False
    file_name = log_dir + '/samples/samples_0~200.json'
    if os.path.exists(file_name):
        with open(file_name) as file:
            data = json.load(file)
        if len(data) >= index:
            return_value = True
    return return_value


def plot_fig(index, log_dir, max_sample_nums):
    ###############################################################
    generation = []
    best_value_list = []
    all_best_value = float('-inf')
    best_alg = None

    file_name = log_dir + '/samples/samples_0~200.json'

    with open(file_name) as file:
        data = json.load(file)

    for i in range(index):
        individual = data[i]
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

    fig = plt.Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"Result display", fontdict=font)

    ax.plot(generation, best_value_list, color='tab:blue', marker='o')
    ax.set_xlabel('Samples', fontdict=font)
    ax.set_ylabel('Current best objective', fontdict=font)
    ax.grid(True)

    if len(generation) <= max_sample_nums:
        ax.set_xticks(np.arange(0, max_sample_nums + 1, 1))
    else:
        ax.set_xticks(np.arange(0, len(generation) + 1, 1))

    ###############################################################

    return fig, best_alg, all_best_value


def display_plot(index):
    global figures

    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig = figures[index]

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    value_label.config(text=f"{index + 1} samples")

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
    link_doc = ttk.Button(top_frame, image=photoimage_doc, command=open_doc_link, bootstyle="info")
    link_doc.pack(side=ttk.LEFT, padx=3)
    link_git = ttk.Button(top_frame, image=photoimage_git, command=open_github_link, bootstyle="info")
    link_git.pack(side=ttk.LEFT, padx=3)
    link_web = ttk.Button(top_frame, image=photoimage_web, command=open_website_link, bootstyle="info")
    link_web.pack(side=ttk.LEFT, padx=3)
    link_qq = ttk.Button(top_frame, image=photoimage_qq, command=open_qq_link, bootstyle="info")
    link_qq.pack(side=ttk.LEFT, padx=3)

    bottom_frame = ttk.Frame(root)
    bottom_frame.pack(fill='both', expand=True)
    left_frame = ttk.Frame(bottom_frame)
    left_frame.grid(row=0, column=0, sticky="nsew")
    ttk.Separator(bottom_frame, orient='vertical', bootstyle="secondary").grid(row=0, column=1, sticky="ns")
    right_frame = ttk.Frame(bottom_frame)
    right_frame.grid(row=0, column=2, sticky="nsew")

    bottom_frame.grid_rowconfigure(0, weight=1)
    bottom_frame.grid_columnconfigure(0, weight=2)
    bottom_frame.grid_columnconfigure(1, weight=1)
    bottom_frame.grid_columnconfigure(2, weight=30)

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
    problem_param_frame = ttk.Labelframe(container_frame_2, text="online_bin_packing", bootstyle="warning")
    problem_param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    ############

    algo_listbox = tk.Listbox(algo_frame, height=6, bg='white', selectbackground='lightgray', font=('Comic Sans MS', 12))
    algo_listbox.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    default_method_index = None
    for _, dict_name, _ in os.walk('../llm4ad/method'):
        for name in dict_name:
            if name != '__pycache__':
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

    root.mainloop()

    ##########################################################