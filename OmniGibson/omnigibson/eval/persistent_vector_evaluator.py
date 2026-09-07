"""Reuse an Isaac application across chunks without changing the baseline evaluator."""

import gc
import logging
import os
from signal import SIGINT, default_int_handler, signal

from omegaconf import OmegaConf

import omnigibson as og
from omnigibson.eval.evaluator import resolve_instance_ids
from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES, seed_everything
from omnigibson.eval.vector_evaluator import VectorChunkEvaluator, _close_video_writer
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


class ReusableVectorChunkEvaluator(VectorChunkEvaluator):
    def close(self):
        """Release task resources, leaving application shutdown to the worker."""
        if self._closed:
            return
        self._closed = True
        for writer in self.video_writers:
            _close_video_writer(writer)
        self.video_writers = [None] * self.num_envs
        self.vector_env.close()
        # The current websocket client has no public close method. Explicitly
        # close its connection before dropping it to avoid per-task threads.
        client = self.policy.policy
        if client is not None and client._ws is not None:
            client._ws.close()
            client._ws = None


class PersistentVectorEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.evaluator = None

    def run_request(self, task_name, instance_indices, output_dir):
        if task_name not in TASK_NAMES_TO_INDICES:
            raise ValueError(f"Unknown 2026 challenge task: {task_name}")
        instance_ids = resolve_instance_ids(task_name, instance_indices, mode=str(self.cfg.mode))
        if self.evaluator is not None and self.evaluator.task_name != task_name:
            previous_task = self.evaluator.task_name
            app = og.app
            self.evaluator.close()
            self.evaluator = None
            # The baseline constructor registers a bound signal handler. Drop
            # that reference before collecting old environments and tensors.
            signal(SIGINT, default_int_handler)
            gc.collect()
            og.clear()
            gc.collect()
            assert og.app is app, "Task switching must preserve the Isaac application"
            logger.info(
                "Persistent task switch: pid=%s %s -> %s; Isaac app retained", os.getpid(), previous_task, task_name
            )

        if self.evaluator is None:
            seed_everything(int(self.cfg.seed))
            cfg = OmegaConf.merge(
                self.cfg,
                {"task": {"name": task_name, "id": TASK_NAMES_TO_INDICES[task_name]}, "output_dir": output_dir},
            )
            self.evaluator = ReusableVectorChunkEvaluator(cfg)
        else:
            self.evaluator.cfg.output_dir = output_dir
            logger.info("Persistent env reuse: pid=%s task=%s indices=%s", os.getpid(), task_name, instance_indices)
        return self.evaluator.run([int(instance_id) for instance_id in instance_ids], rollout_id=0)

    def close(self):
        try:
            if self.evaluator is not None:
                self.evaluator.close()
                self.evaluator = None
        finally:
            og.shutdown()
