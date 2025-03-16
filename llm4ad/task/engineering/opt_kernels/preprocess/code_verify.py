import os
import torch
import random
import shutil
import importlib
import numpy as np

class CodeVerify(object):
    def __init__(self, verify_counts=5, res_path=None, keep_temp=False, specifier=None):
        self.verify_counts = verify_counts
        assert res_path is not None, "Please provide a path to save the results."
        self.res_path = res_path
        self.keep_temp = keep_temp
        if specifier is not None:
            self.res_path = os.path.join(res_path, specifier)
        os.makedirs(self.res_path, exist_ok=True)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def load_module_from_path(code_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, spec

    @staticmethod
    def write_multiple_files_to_multiple_paths(file_paths, contents):
        for file_path, content in zip(file_paths, contents):
            CodeVerify.write_file_to_path(file_path, content)

    @staticmethod
    def write_file_to_path(file_path, content):
        with open(file_path, "w") as f:
            f.write(content)

    @staticmethod
    def clear_multiple_files(file_paths):
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

    @staticmethod
    def clear_dir(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
