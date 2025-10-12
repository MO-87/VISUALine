import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from visualine.core.config_loader import BaseConfigLoader
from visualine.core.node_base import NodeBase
from visualine.core.task_executer import TaskExecuter
from visualine.utils.file_io import FFmpegError, VideoProcessor

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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
        node_configs = config.get("pipeline", [])

        if not node_configs:
            logger.warning("Pipeline configuration is empty.")
            return

        self._pipeline = [self._create_node_instance(cfg) for cfg in node_configs]
        logger.info(f"Loaded {len(self._pipeline)} nodes successfully.")
        for node in self._pipeline:
            logger.debug(f" -> Node: {repr(node)}")

    def run(self, input_path: Path, output_path: Path) -> None:
        """Runs the pipeline on either a single image, a directory of images, or a video file."""
        if not self._pipeline:
            logger.error("No pipeline loaded — cannot run.")
            return

        file_suffix = input_path.suffix.lower()

        if input_path.is_dir():
            logger.info(f"Detected image directory input: {input_path}")
            self._run_image_batch(input_path, output_path)
        elif file_suffix in SUPPORTED_IMAGE_EXTENSIONS:
            logger.info(f"Detected single image input: {input_path}")
            self._run_image(input_path, output_path)
        elif file_suffix in SUPPORTED_VIDEO_EXTENSIONS:
            logger.info(f"Detected video input: {input_path}")
            self._run_video(input_path, output_path)
        else:
            raise ValueError(f"Unsupported file type: '{file_suffix}'.")

    def _run_image(self, input_image_path: Path, output_image_path: Path) -> None:
        """Runs the pipeline on a single image."""
        try:
            image_bgr = cv2.imread(str(input_image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot load image: {input_image_path}")
            
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            if self._executer._use_cuda:
                tensor = tensor.cuda(non_blocking=True)

            processed_tensor = self._executer.execute_batch(self._pipeline, tensor)

            final_image_rgb = processed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            final_image = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), final_image)

            logger.info(f"Image pipeline completed successfully. Output saved to: {output_image_path}")
        except Exception as e:
            logger.critical(f"Image pipeline run failed: {e}", exc_info=True)

    def _run_image_batch(self, input_dir: Path, output_dir: Path) -> None:
        """Runs the pipeline on a directory of images using batched GPU-optimized processing."""
        try:
            image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
            if not image_paths:
                raise FileNotFoundError(f"No supported images found in directory: {input_dir}")

            logger.info(f"Running image batch pipeline on {len(image_paths)} images (batch={self._batch_size})...")
            output_dir.mkdir(parents=True, exist_ok=True)

            batch_buffer, batch_names = [], []
            for idx, img_path in enumerate(image_paths, 1):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.warning(f"Skipping unreadable image: {img_path}")
                    continue
                
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                batch_buffer.append(img)
                batch_names.append(img_path.name)

                if len(batch_buffer) >= self._batch_size:
                    self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                    batch_buffer, batch_names = [], []
                
                if idx % 50 == 0 or idx == len(image_paths):
                    logger.info(f"Progress: {idx}/{len(image_paths)} images processed.")

            if batch_buffer:
                self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)

            logger.info(f"Image directory pipeline completed successfully. Outputs saved to: {output_dir}")
        except Exception as e:
            logger.critical(f"Image batch pipeline run failed: {e}", exc_info=True)

    def _process_and_save_image_batch(self, images: List[np.ndarray], names: List[str], output_dir: Path) -> None:
        """Processes and saves a batch of images."""
        try:
            batch_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
            if self._executer._use_cuda:
                batch_tensor = batch_tensor.cuda(non_blocking=True)

            processed_batch = self._executer.execute_batch(self._pipeline, batch_tensor)
            final_batch_rgb = processed_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            for name, img_rgb in zip(names, final_batch_rgb):
                out_path = output_dir / name
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), img_bgr)

            del batch_tensor, processed_batch, final_batch_rgb
            if self._executer._use_cuda:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Batch image processing failed: {e}", exc_info=True)

    def _run_video(self, input_video_path: Path, output_video_path: Path) -> None:
        """Runs the pipeline on a video file using batched GPU-optimized processing."""
        try:
            with VideoProcessor(input_video_path) as processor:
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    raise FileNotFoundError(f"Cannot open video: {input_video_path}")

                fps, width, height, total_frames = (
                    processor.get_framerate(),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                )

                temp_video_path = output_video_path.with_name(f"{output_video_path.stem}_temp_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

                audio_path = processor.extract_audio()
                frame_buffer, processed_frames = [], 0

                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        ## flushing remaining frames if any
                        if frame_buffer:
                            self._process_and_write_batch(frame_buffer, out)
                            processed_frames += len(frame_buffer)
                        break
                    
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(frame)

                    if len(frame_buffer) >= self._batch_size:
                        self._process_and_write_batch(frame_buffer, out)
                        processed_frames += len(frame_buffer)
                        frame_buffer = []
                    
                    if (processed_frames > 0) and (processed_frames % 50 == 0):
                        logger.info(f"Progress: {processed_frames}/{total_frames} frames processed.")

                cap.release()
                out.release()
                logger.info(f"Finished processing {processed_frames} frames.")

                logger.info("Merging audio back into the final video...")
                processor.recombine_video(
                    output_path=output_video_path,
                    temp_video_path=temp_video_path,
                    audio_path=audio_path
                )
                temp_video_path.unlink(missing_ok=True)

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
            final_batch_rgb = processed_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            for img_rgb in final_batch_rgb:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)

            del batch_tensor, processed_batch, final_batch_rgb
            if self._executer._use_cuda:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Batch video processing failed: {e}", exc_info=True)

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
            raise ImportError(
                f"Could not load class '{class_name}' from module '{full_module_path}'. "
                f"Check your pipeline configuration. Original error: {e}"
            )