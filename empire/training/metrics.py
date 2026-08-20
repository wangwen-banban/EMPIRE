"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

import time
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import jsonlines
import numpy as np
import torch
import wandb

from torch.utils.tensorboard import SummaryWriter

from empire.utils.run_logger import initialize_run_logger

# Initialize RunLogger =>> Wraps `logging.Logger`
run_logger = initialize_run_logger(__name__)


class TensorBoardLogHandler(logging.Handler):
    """Forward Python logs into TensorBoard text events (rank0 only by logger level)."""

    def __init__(
        self,
        writer: SummaryWriter,
        get_global_step,
        tag_prefix: str = "Logs",
        level: int = logging.INFO,
    ) -> None:
        super().__init__(level=level)
        self._writer = writer
        self._get_global_step = get_global_step
        self._tag_prefix = tag_prefix

    def emit(self, record: logging.LogRecord) -> None:
        try:
            step = int(self._get_global_step())
        except Exception:
            step = 0

        try:
            msg = record.getMessage()
            tag = f"{self._tag_prefix}/{record.levelname}"
            text = f"[{record.name}] {msg}"
            self._writer.add_text(tag, text, global_step=step)
        except Exception:
            # Never break training due to logging.
            return


def _install_tensorboard_log_handler(writer: SummaryWriter, get_global_step) -> None:
    """Install a root logger handler to mirror logs to TensorBoard (idempotent)."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, TensorBoardLogHandler) and getattr(h, "_writer", None) is writer:
            return

    root.addHandler(TensorBoardLogHandler(writer=writer, get_global_step=get_global_step))


def _install_file_log_handler(log_path: Path) -> None:
    """Deprecated: file logging is installed early in `scripts/train.py`.

    Kept as a no-op for backward compatibility.
    """
    return


# === Define Tracker Interface ===
class Tracker(Protocol):
    def write_hyperparameters(self) -> None: ...

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None: ...

    def finalize(self) -> None: ...


# === Individual Tracker Definitions ===
class JSONLinesTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @run_logger.rank_zero_only
    def write_hyperparameters(self) -> None:
        with jsonlines.open(self.run_dir / "run-metrics.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @run_logger.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return

class TensorboardTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.hparams = hparams
        self.writer = SummaryWriter(log_dir=str(run_dir / Path(run_id)))

        # Final metrics for add_hparams (can be passed at the end of training)
        self.final_metrics: Dict[str, float] = {}

    def _sanitize_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """Convert hyperparameters to basic types supported by TensorBoard.

        TensorBoard only accepts int/float/str/bool/torch.Tensor;
        other types are converted to strings here to avoid ValueError.
        """
        clean: Dict[str, Any] = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    @run_logger.rank_zero_only
    def write_hyperparameters(self, metrics: Dict[str, float] = None) -> None:
        """
        Record hyperparameters. Recommended to call after training ends, passing the final key metrics.
        """
        if metrics is None:
            metrics = self.final_metrics  # use the cached final metrics

        # Filter/convert to types supported by TensorBoard
        clean_hparams = self._sanitize_hparams(self.hparams)

        # Method 1: officially recommended - use add_hparams (shows a dedicated table in TensorBoard)
        self.writer.add_hparams(
            hparam_dict=clean_hparams,
            metric_dict=metrics,
            # name and global_step are optional, for distinguishing multiple experiment runs
        )

        # Method 2 (optional): keep the original per-key add_text approach (easier to view in the Text tab)
        for k, v in clean_hparams.items():
            self.writer.add_text(f"hparams/{k}", str(v))

        self.writer.flush()

    @run_logger.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        """
        Record scalar metrics
        """
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), global_step)  # ensure it is a float
        # No need to flush every time; TensorBoard refreshes periodically

    @run_logger.rank_zero_only
    def set_final_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Call before training ends to cache final metrics for write_hyperparameters
        """
        self.final_metrics = metrics

    @run_logger.rank_zero_only
    def finalize(self, success: bool = True) -> None:
        """
        Close the writer. Recommended to call at the end of training (whether successful or on error).
        """
        if success and self.final_metrics:
            # If there are final metrics and write_hyperparameters was not called before, write once more
            self.write_hyperparameters()
        
        self.writer.close()

class WeightsBiasesTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "prismatic",
        entity: Optional[str] = None,
        group: str = "align",
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Get W&B-Specific Initialization Parameters
        self.project, self.entity, self.group, self.wandb_dir = project, entity, group, self.run_dir

        # Call W&B.init()
        self.initialize()

    @run_logger.rank_zero_only
    def initialize(self) -> None:
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    @run_logger.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @run_logger.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if run_logger.is_rank_zero():
            wandb.finish()

        # A job gets 210 seconds to get its affairs in order
        time.sleep(210)


# === Core Metrics Container :: Initializes Trackers => Compiles/Pushes Metrics ===


class Metrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        stage: str,
        wandb_project: str = "prismatic",
        wandb_entity: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
    ) -> None:
        self.run_id, self.run_dir, self.hparams, self.stage = run_id, run_dir, hparams, stage

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity, group=self.stage
                )
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step, self.start_time, self.step_start_time = 0, time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self, *, global_step: Optional[int] = None, lr: Optional[float] = None, update_step_time: bool = False, **kwargs
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        # For all other variables --> only track on rank zero!
        if not run_logger.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                if key in self.state:
                    self.state[key].append(value.detach())
                else:
                    if isinstance(value, torch.Tensor):
                        self.other_state[key].append(value.detach())
                    else:
                        self.other_state[key].append(value)

    @run_logger.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        step_time, lr = np.mean(list(self.state["step_time"])), self.state["lr"][-1]
        status = self.get_status(loss)

        # Fire to Trackers
        prefix = self.stage.capitalize()
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Loss": loss,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
            },
        )
        return status

    def finalize(self) -> str:
        for tracker in self.trackers:
            tracker.finalize()


class VLAMetrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        wandb_project: str = "empire",
        wandb_entity: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: Optional[int] = None,
        resume_epoch: Optional[int] = None,
        log_dir: Path = None,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams
        if isinstance(run_dir, str):
            self.run_dir = Path(run_dir)
        self

        # Initialize Trackers
        self.trackers = []
        self._tb_tracker: Optional[TensorboardTracker] = None
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, self.run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, self.run_dir, hparams, project=wandb_project, entity=wandb_entity, group="vla-train"
                )
            elif tracker_type == "tensorboard":
                tracker = TensorboardTracker(run_id, self.run_dir, hparams)
                self._tb_tracker = tracker
            else:
                raise ValueError(f"Tracker with type `{tracker_type}` is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch
        self.start_time, self.step_start_time = time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }
        self.other_state = defaultdict(lambda: deque(maxlen=window_size))

        if run_logger.is_rank_zero():
            # Mirror python logs into TensorBoard (rank0 only)
            if self._tb_tracker is not None:
                _install_tensorboard_log_handler(self._tb_tracker.writer, get_global_step=lambda: self.global_step)

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> Backbone LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> Backbone LR :: {lr:.6f} - Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not run_logger.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                if key in self.state:
                    self.state[key].append(value.detach())
                else:
                    if isinstance(value, torch.Tensor):
                        self.other_state[key].append(value.detach())
                    else:
                        self.other_state[key].append(torch.tensor(value, dtype=torch.float32))

    @run_logger.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        step_time = np.mean(list(self.state["step_time"]))
        lr = self.state["lr"][-1]

        # stage/scheduler compatibility:
        # - multi-group scheduler logs action_decay_lr/action_no_decay_lr
        # - single-group scheduler may not log action-specific LR at all
        # In the latter case, fall back to backbone lr for stable logging.
        action_lr_hist = self.other_state.get("action_decay_lr", None)
        if action_lr_hist is None or len(action_lr_hist) == 0:
            action_model_lr = lr
        else:
            action_lr_last = action_lr_hist[-1]
            if isinstance(action_lr_last, torch.Tensor):
                action_model_lr = action_lr_last.item() if action_lr_last.numel() == 1 else action_lr_last.mean().item()
            else:
                action_model_lr = float(action_lr_last)

        status = self.get_status(loss)
        # Additional metrics from other_state
        additional_metrics = {
            f"Other/{key}": torch.stack(list(value)).mean().item()
            for key, value in self.other_state.items()
        }

        # Log metrics
        prefix = "VLA Train"
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Epoch": self.epoch,
                f"{prefix}/Loss": loss,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Backbone Learning Rate": lr,
                f"{prefix}/Action Model Learning Rate": action_model_lr,
                f"{prefix}/Step Time": step_time,
                **additional_metrics,
            },
        )
        return status

    def finalize(self) -> str:
        for tracker in self.trackers:
            tracker.finalize()
