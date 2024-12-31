from __future__ import annotations

import json
import os
from threading import Lock

from ...base import Function


class ProfilerBase:
    _num_samples = 0

    def __init__(self, log_dir: str | None = None, initial_num_samples=0, log_style='complex'):
        assert log_style in ['simple', 'complex']
        self.__class__._num_samples = initial_num_samples
        self._log_dir = log_dir
        self._log_style = log_style
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = float('-inf')
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0

        if log_dir:
            self._samples_json_dir = os.path.join(log_dir, 'samples')
            os.makedirs(self._samples_json_dir, exist_ok=True)

        # lock for multi-thread invoking self.register_function(...)
        self._register_function_lock = Lock()

    def register_function(self, function: Function, *, resume_mode=False):
        """Record an obtained function. This is a synchronized function.
        """
        try:
            self._register_function_lock.acquire()
            self.__class__._num_samples += 1
            self._record_and_verbose(function, resume_mode=resume_mode)
            self._write_json(function)
        finally:
            self._register_function_lock.release()

    def finish(self):
        pass

    def get_logger(self):
        pass

    def resume(self, *args, **kwargs):
        pass

    def _write_json(self, function: Function):
        if not self._log_dir:
            return

        sample_order = self.__class__._num_samples
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(function)
        score = function.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._samples_json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _record_and_verbose(self, function, *, resume_mode=False):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        # update best function
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_function = function
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self.__class__._num_samples

        if not resume_mode:
            # log attributes of the function
            if self._log_style == 'complex':
                print(f'================= Evaluated Function =================')
                print(f'{function_str}')
                print(f'------------------------------------------------------')
                print(f'Score        : {str(score)}')
                print(f'Sample time  : {str(sample_time)}')
                print(f'Evaluate time: {str(evaluate_time)}')
                print(f'Sample orders: {str(self.__class__._num_samples)}')
                print(f'------------------------------------------------------')
                print(f'Current best score: {self._cur_best_program_score}')
                print(f'======================================================\n')
            else:
                if score is None:
                    print(f'Sample{self.__class__._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}')
                else:
                    print(f'Sample{self.__class__._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
