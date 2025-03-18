import tkinter as tk

def show_value():
    # 直接读取变量值，无需调用 entry.get()
    print("当前输入的值是:", entry_var.get())

root = tk.Tk()

# 1. 创建 StringVar 变量
entry_var = tk.StringVar()

# 2. 绑定变量到 Entry 控件
entry = tk.Entry(root, textvariable=entry_var)
entry.pack(pady=10)

# 按钮点击时打印变量值
btn = tk.Button(root, text="显示输入内容", command=show_value)
btn.pack()

root.mainloop()