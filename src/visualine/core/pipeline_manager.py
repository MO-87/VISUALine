import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from visualine.core.config_loader import BaseConfigLoader
from visualine.core.node_base import NodeBase
from visualine.utils.file_io import VideoProcessor, FFmpegError
from visualine.core.task_executer import TaskExecuter

logger = logging.getLogger(__name__)


class PipelineManager:
    """High-level orchestrator for building and running a visual processing pipeline."""

    def __init__(self, config_loader: BaseConfigLoader, node_namespace: str = "visualine.nodes"):
        self._config_loader = config_loader
        self._node_namespace = node_namespace
        self._pipeline: List[NodeBase] = []
        self._executer = TaskExecuter()
        self._batch_size = 8  ## adjustable batch size for optimized throughput

    def load_pipeline(self, pipeline_config_path: Path) -> None:
        """Loads pipeline configuration and builds node chain."""
        logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")
        config = self._config_loader.load(pipeline_config_path)

        pipeline_nodes_config = config.get("pipeline", [])
        if not pipeline_nodes_config:
            logger.warning("Pipeline configuration is empty.")
            return

        self._pipeline = [self._create_node_instance(nc) for nc in pipeline_nodes_config]
        logger.info(f"Loaded pipeline with {len(self._pipeline)} nodes.")
        for node in self._pipeline:
            logger.debug(f" -> Node: {repr(node)}")

    def run(self, input_video_path: Path, output_video_path: Path) -> None:
        """Runs the pipeline on a video file using batched GPU-optimized processing."""
        if not self._pipeline:
            logger.error("No pipeline loaded — cannot run.")
            return

        logger.info(f"Running pipeline on '{input_video_path}' (batch={self._batch_size})...")

        try:
            with VideoProcessor(input_video_path) as processor:
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    raise FileNotFoundError(f"Cannot open video: {input_video_path}")

                fps = processor.get_framerate()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

                audio_path = processor.extract_audio()

                frame_buffer = []
                processed_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        ## flush remaining frames if any
                        if frame_buffer:
                            self._process_and_write_batch(frame_buffer, out)
                        break

                    frame_buffer.append(frame)
                    if len(frame_buffer) >= self._batch_size:
                        self._process_and_write_batch(frame_buffer, out)
                        frame_buffer = []

                    processed_frames += 1
                    if processed_frames % 50 == 0:
                        logger.info(f"Progress: {processed_frames}/{total_frames} frames processed.")

                cap.release()
                out.release()

                ## merging audio back if exists
                if audio_path and audio_path.exists():
                    processor._run_command([
                        "ffmpeg", "-y",
                        "-i", str(output_video_path),
                        "-i", str(audio_path),
                        "-c:v", "copy", "-c:a", "aac", "-shortest",
                        str(output_video_path)
                    ])

            logger.info(f"Pipeline completed successfully. Output saved to: {output_video_path}")

        except (FFmpegError, Exception) as e:
            logger.critical(f"Pipeline run failed: {e}", exc_info=True)

    def _process_and_write_batch(self, frames: List[np.ndarray], out: cv2.VideoWriter) -> None:
        """Processes and writes a batch of frames using the TaskExecuter."""
        try:
            batch_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
            if self._executer._use_cuda:
                batch_tensor = batch_tensor.cuda(non_blocking=True)

            processed_batch = self._executer.execute_batch(self._pipeline, batch_tensor)
            final_batch = processed_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            for f in final_batch:
                out.write(f)

        except Exception as e:
            logger.error(f"Batch processing failed, falling back to per-frame mode: {e}", exc_info=True)
            ## fallback to CPU per-frame processing
            for f in frames:
                for node in self._pipeline:
                    f = self._executer.execute_frame(node, f)
                out.write(f)

    def _create_node_instance(self, node_config: Dict[str, Any]) -> NodeBase:
        """Dynamically loads a node class based on the provided config."""
        class_path = node_config.get("class")
        if not class_path:
            raise ValueError("Node configuration missing 'class' key.")

        try:
            module_path, class_name = class_path.rsplit('.', 1)
            full_module_path = f"{self._node_namespace}.{module_path}"
            node_module = importlib.import_module(full_module_path)
            node_class = getattr(node_module, class_name)

            if not issubclass(node_class, NodeBase):
                raise TypeError(f"{class_name} is not a subclass of NodeBase.")

            params = node_config.get("params", {})
            return node_class(config=params)
        
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load node class '{class_path}': {e}")