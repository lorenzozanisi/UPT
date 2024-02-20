from datetime import datetime

from callbacks.base.periodic_callback import PeriodicCallback
from utils.formatting_util import seconds_to_duration_str


class ProgressCallback(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = None
        self._last_log_time = None
        self._last_log_samples = None

    def _before_training(self, **kwargs):
        self._start_time = self._last_log_time = datetime.now()
        self._last_log_samples = 0

    # noinspection PyMethodOverriding
    def _periodic_callback(self, trainer, interval_type, **_):
        if trainer.end_checkpoint.epoch is not None:
            total_updates = trainer.end_checkpoint.epoch * self.update_counter.updates_per_epoch
        elif trainer.end_checkpoint.update is not None:
            total_updates = trainer.end_checkpoint.update
        elif trainer.end_checkpoint.sample is not None:
            total_updates = self.update_counter.cur_checkpoint.sample // self.update_counter.effective_batch_size
        else:
            raise NotImplementedError

        self.logger.info("------------------")
        if interval_type == "epoch":
            self.logger.info(
                f"Epoch {self.update_counter.cur_checkpoint.epoch}/{trainer.end_checkpoint.epoch} "
                f"({self.update_counter.cur_checkpoint})"
            )
        elif interval_type == "update":
            self.logger.info(
                f"Update {self.update_counter.cur_checkpoint.update}/{total_updates} "
                f"({self.update_counter.cur_checkpoint})"
            )
        elif interval_type == "sample":
            self.logger.info(
                f"Sample {self.update_counter.cur_checkpoint.sample}/{trainer.end_checkpoint.sample} "
                f"({self.update_counter.cur_checkpoint})"
            )
        else:
            raise NotImplementedError
        progress = self.update_counter.cur_checkpoint.update / total_updates
        now = datetime.now()
        estimated_duration = (now - self._start_time) / progress
        seconds_since_last_log = (now - self._last_log_time).total_seconds()
        samples_since_last_log = self.update_counter.cur_checkpoint.sample - self._last_log_samples
        updates_since_last_log = samples_since_last_log // self.update_counter.effective_batch_size
        self.logger.info(
            f"ETA: {(self._start_time + estimated_duration).strftime('%m.%d %H.%M.%S')} "
            f"estimated_duration: {seconds_to_duration_str(estimated_duration.total_seconds())} "
            f"time_since_last_log: {seconds_to_duration_str(seconds_since_last_log)} "
            f"time_per_update: {seconds_to_duration_str(seconds_since_last_log / updates_since_last_log)} "
        )
        self.writer.add_scalar(f"profiling/time_since_last_log/{interval_type}", seconds_since_last_log)
        self._last_log_time = now
        self._last_log_samples = self.update_counter.cur_checkpoint.sample
