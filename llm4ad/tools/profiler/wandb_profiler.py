from __future__ import annotations

import sys

from llm4ad.base import Function
from llm4ad.tools.profiler.profile import ProfilerBase

try:
    import wandb
except:
    pass


class WandBProfiler(ProfilerBase):
    # _num_samples = 0

    def __init__(self,
                 wandb_project_name: str,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 **wandb_init_kwargs):
        """
        Args:
            wandb_project_name : the project name in which you sync your results.
            log_dir            : folder path for tensorboard log files.
            wandb_init_kwargs  : args used to init wandb project, such as name='funsearch_run1', group='funsearch'.
            log_style          : the output style in the terminal. Option in ['complex', 'simple'].
        """

        super().__init__(log_dir=log_dir, initial_num_samples=initial_num_samples, log_style=log_style)
        self._wandb_project_name = wandb_project_name

        # for MacOS and Linux
        if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
            setting = wandb.Settings(start_method='fork')
            self._logger = wandb.init(
                project=self._wandb_project_name,
                settings=setting,
                **wandb_init_kwargs
            )
        else:  # for Windows
            wandb.setup()
            self._logger = wandb.init(
                project=self._wandb_project_name,
                **wandb_init_kwargs
            )

    def get_logger(self):
        return self._logger

    def register_function(self, function: Function, *, resume_mode=False):
        """Record an obtained function. This is a synchronized function.
        """
        try:
            self._register_function_lock.acquire()
            self.__class__._num_samples += 1
            self._record_and_verbose(function, resume_mode=resume_mode)
            self._write_wandb()
            self._write_json(function)
        finally:
            self._register_function_lock.release()

    def _write_wandb(self, *args, **kwargs):
        self._logger.log(
            {
                'Best Score of Function': self._cur_best_program_score
            },
            step=self.__class__._num_samples
        )
        self._logger.log(
            {
                'Valid Function Num': self._evaluate_success_program_num,
                'Invalid Function Num': self._evaluate_failed_program_num
            },
            step=self.__class__._num_samples
        )
        self._logger.log(
            {
                'Total Sample Time': self._tot_sample_time,
                'Total Evaluate Time': self._tot_evaluate_time
            },
            step=self.__class__._num_samples
        )

    def finish(self):
        wandb.finish()
