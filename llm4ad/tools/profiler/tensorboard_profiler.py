from __future__ import annotations

import json
import os
from threading import Lock

from llm4ad.base import Function
from llm4ad.tools.profiler.profile import ProfilerBase

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass


class TensorboardProfiler(ProfilerBase):

    def __init__(self,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex'):
        """
        Args:
            log_dir  : folder path for tensorboard log files.
            log_style: the output style in the terminal. Option in ['complex', 'simple']
        """
        super().__init__(log_dir=log_dir, initial_num_samples=initial_num_samples, log_style=log_style)

        # summary writer instance for Tensorboard
        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

    def get_logger(self):
        return self._writer

    def register_function(self, function: Function, *, resume_mode=False):
        """Record an obtained function. This is a synchronized function.
        """
        try:
            self._register_function_lock.acquire()
            self.__class__._num_samples += 1
            self._record_and_verbose(function, resume_mode=resume_mode)
            self._write_tensorboard()
            self._write_json(function)
        finally:
            self._register_function_lock.release()

    def finish(self):
        if self._log_dir:
            self._writer.close()

    def _write_tensorboard(self, *args, **kwargs):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self.__class__._num_samples
        )
