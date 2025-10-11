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

logger = logging.getLogger(__name__)


class PipelineManager:
    """Efficient in-memory pipeline runner with GPU acceleration and batch optimization."""

    def __init__(self, config_loader: BaseConfigLoader, node_namespace: str = "visualine.nodes"):
        self._config_loader = config_loader
        self._node_namespace = node_namespace
        self._pipeline: List[NodeBase] = []
        self._use_cuda = torch.cuda.is_available()
        self._batch_size = 8  ## adjustable batch size for optimized throughput

        if self._use_cuda:
            logger.info(f"GPU acceleration enabled ({torch.cuda.get_device_name(0)}).")
        else:
            logger.info("GPU not available. Running on CPU.")

    def load_pipeline(self, pipeline_config_path: Path) -> None:
        """Loads pipeline configuration and builds node chain."""
        logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")
        config = self._config_loader.load(pipeline_config_path)

        pipeline_nodes_config = config.get("pipeline", [])
        if not pipeline_nodes_config:
            logger.warning("Pipeline configuration is empty.")
            self._pipeline = []
            return

        self._pipeline = [self._create_node_instance(nc) for nc in pipeline_nodes_config]
        logger.info(f"Loaded pipeline with {len(self._pipeline)} nodes.")
        for node in self._pipeline:
            logger.debug(f" -> Node: {repr(node)}")

    def run(self, input_video_path: Path, output_video_path: Path) -> None:
        """Runs the loaded pipeline on a video in-memory, processing in batches for speed."""
        if not self._pipeline:
            logger.error("No pipeline loaded — cannot run.")
            return

        logger.info(f"Running batch-optimized pipeline on '{input_video_path}'...")

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
                frame_idx = 0

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

                    frame_idx += 1

                    if frame_idx % 50 == 0 or frame_idx == total_frames:
                        logger.info(f"Progress: {frame_idx}/{total_frames} frames processed.")

                cap.release()
                out.release()

                ## merging audio if exists
                if audio_path and audio_path.exists():
                    processor._run_command([
                        "ffmpeg", "-y",
                        "-i", str(output_video_path),
                        "-i", str(audio_path),
                        "-c:v", "copy", "-c:a", "aac", "-shortest",
                        str(output_video_path)
                    ])

            logger.info(f"Pipeline completed successfully (GPU={self._use_cuda}). Output: {output_video_path}")

        except (FFmpegError, Exception) as e:
            logger.critical(f"Pipeline run failed: {e}", exc_info=True)

    def _process_and_write_batch(self, frames: List[np.ndarray], out: cv2.VideoWriter) -> None:
        """
        Processes a batch of frames through all nodes, GPU-optimized if available.
        """
        try:
            ## moving batch to GPU tensor if supported
            if self._use_cuda:
                batch_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().cuda(non_blocking=True)
            else:
                batch_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()

            ## process through each node
            for node in self._pipeline:
                if hasattr(node, "use_torch") and node.use_torch:
                    batch_tensor = node.process(batch_tensor)
                else:
                    ## convert to numpy batch for OpenCV-based nodes
                    batch_np = batch_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    batch_np = np.stack([node.process(f) for f in batch_np])
                    batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float()
                    if self._use_cuda:
                        batch_tensor = batch_tensor.cuda(non_blocking=True)

            ## back to CPU numpy for writing
            final_batch = batch_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            ## write processed frames
            for frame in final_batch:
                out.write(frame)

        except Exception as e:
            logger.error(f"Error while processing batch: {e}", exc_info=True)
            ## fallback to CPU per-frame processing
            for f in frames:
                for node in self._pipeline:
                    f = self._process_frame_with_node(node, f)
                out.write(f)

    def _process_frame_with_node(self, node: NodeBase, frame: Any) -> Any:
        """
        Processes a frame through a node, auto-handling CPU/GPU conversion if necessary.
        """
        try:
            ## handling PyTorch-based nodes
            if hasattr(node, "use_torch") and node.use_torch:
                if isinstance(frame, np.ndarray):
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
                    if self._use_cuda:
                        frame_tensor = frame_tensor.cuda(non_blocking=True)
                else:
                    frame_tensor = frame  ## assuming already on GPU

                result = node.process(frame_tensor)

                ## converting back to numpy if torch tensor
                if torch.is_tensor(result):
                    result = result.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

                return result

            ## otherwise assume OpenCV-based node
            return node.process(frame)

        except Exception as e:
            logger.error(f"Error in node '{node.__class__.__name__}': {e}")
            return frame

    def _create_node_instance(self, node_config: Dict[str, Any]) -> NodeBase:
        """Dynamically loads a node class from config."""
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