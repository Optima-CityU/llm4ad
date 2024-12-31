from __future__ import annotations

import copy
import json
import os.path

from tqdm.auto import tqdm

from .profiler import RandSampleProfiler
from .randsample import RandSample
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_all_samples_and_scores(path):
    path = os.path.join(path, 'samples')

    def path_to_int(path):
        num = int(path.split('.')[0].split('_')[1])
        return num

    all_func = []
    all_score = []
    dirs = list(os.listdir(path))
    dirs = sorted(dirs, key=path_to_int)
    max_o = path_to_int(dirs[-1])

    for dir in dirs:
        file_name = os.path.join(path, dir)
        with open(file_name, 'r') as f:
            sample = json.load(f)
        func = sample['function']
        acc = sample['score'] if sample['score'] else float('-inf')
        all_func.append(func)
        all_score.append(acc)

    return all_func, all_score, max_o


def _resume_text2func(f, s, template_func: Function):
    temp = copy.deepcopy(template_func)
    f = tfpc.text_to_function(f)
    if f is None:
        temp.body = '    pass'
        temp.score = None
        return temp
    else:
        f.score = s
        return f


def _resume_pf(log_path: str, pf: RandSampleProfiler, template_func: Function):
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    print(f'RESUME RandSample: Sample order: {sample_max_order}.', flush=True)
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume RandSample Profiler'):  # noqa
        f, s = funcs[i], scores[i]  # noqa
        f = _resume_text2func(f, s, template_func)
        pf.register_function(f, resume_mode=True)


def resume_randsample(rs: RandSample):
    rs._resume_mode = True
    pf = rs._profiler
    assert pf is not None
    log_path = pf._log_dir

    # resume profiler
    template_func = rs._function_to_evolve
    _resume_pf(log_path, pf, template_func)

    # resume rand sample
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    rs._tot_sample_nums = sample_max_order
