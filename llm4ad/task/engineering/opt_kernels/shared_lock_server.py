# 文件名：cross_file_lock.py
import os
import time
from contextlib import contextmanager

# 使用 portalocker 库（跨平台）
import portalocker

LOCK_FILE = "cross_process.lock"

@contextmanager
def global_file_lock():
    # 打开（或创建）锁文件
    with open(LOCK_FILE, "w") as f:
        try:
            # 请求非阻塞锁（可根据需要调整参数）
            portalocker.lock(f, portalocker.LOCK_EX)
            print(f"Lock acquired by PID: {os.getpid()}")
            yield  # 进入临界区
        finally:
            portalocker.unlock(f)
            print(f"Lock released by PID: {os.getpid()}")
            # os.remove(LOCK_FILE)  # 可选：清理锁文件
