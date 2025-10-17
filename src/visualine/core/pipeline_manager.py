import os
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            self._batch_size = self._auto_select_batch_size()  ## increase/decrease if you have high/low VRAM
            self._max_workers = 1
        else:
            cpu_cores = os.cpu_count() or 4
            self._batch_size = 8
            self._max_workers = min(4, cpu_cores // 2)

    def load_pipeline(self, pipeline_config_path: Path) -> None:
        """Loads pipeline configuration, builds the node chain, and runs setup for each node."""
        logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")
        config = self._config_loader.load(pipeline_config_path)
        node_configs = config.get("pipeline", [])

        if not node_configs:
            logger.warning("Pipeline configuration is empty.")
            return

        if self._pipeline:
            logger.info("Tearing down the existing pipeline before loading a new one...")
            for node in self._pipeline:
                node.teardown()

        self._pipeline = [self._create_node_instance(cfg) for cfg in node_configs]
        logger.info(f"Loaded {len(self._pipeline)} nodes successfully.")
        for node in self._pipeline:
            logger.debug(f" -> Node: {repr(node)}")

        logger.info("Setting up pipeline nodes...")
        device_obj = torch.device(self.device)
        for node in self._pipeline:
            try:
                node.setup(device=device_obj)
            except Exception as e:
                logger.error(f"Failed to setup node {node.node_name}: {e}", exc_info=True)
                raise
        logger.info("All nodes have been set up successfully.")

    def run(self, input_path: Path, output_path: Path) -> None:
        """
        Runs the pipeline on an input file/directory and ensures teardown is called afterward.
        """
        if not self._pipeline:
            logger.error("No pipeline loaded — cannot run.")
            return

        try:
            logger.info(f"Initialized PipelineManager on {self.device.upper()} | "
                        f"batch_size={self._batch_size}, max_workers={self._max_workers}")

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
        finally:
            logger.info("Tearing down pipeline nodes...")
            for node in self._pipeline:
                node.teardown()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Pipeline teardown complete.")

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

    def _safe_execute_batch(self, batch_tensor):
        """Executes a GPU batch safely with adaptive fallback in case of OOM."""
        try:
            return self._executer.execute_batch(self._pipeline, batch_tensor)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._batch_size = max(1, self._batch_size // 2)
            logger.warning(f"[WARNING] GPU OOM detected. Reducing batch size to {self._batch_size}.")
            raise

    def _run_video(self, input_video_path: Path, output_video_path: Path) -> None:
        """Runs the pipeline on a video file using async batched GPU-optimized processing."""
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

                ## init audio extraction in parallel
                audio_thread = threading.Thread(target=processor.extract_audio)
                audio_thread.start()

                output_resolution = (width, height)
                read_queue = queue.Queue(maxsize=4)
                write_queue = queue.Queue()
                stop_signal = object()

                ## prefetch thread (I/O bound)
                def frame_reader():
                    frame_buffer = []
                    while True:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            if frame_buffer:
                                read_queue.put(frame_buffer)
                            read_queue.put(stop_signal)
                            break

                        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_buffer.append(frame)

                        if len(frame_buffer) >= self._batch_size:
                            read_queue.put(frame_buffer)
                            frame_buffer = []

                ## write thread (I/O bound)
                def frame_writer():
                    while True:
                        item = write_queue.get()
                        if item is stop_signal:
                            break
                        for img_rgb in item:
                            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                            out.write(img_bgr)
                        write_queue.task_done()

                reader_thread = threading.Thread(target=frame_reader, daemon=True)
                writer_thread = threading.Thread(target=frame_writer, daemon=True)
                reader_thread.start()
                writer_thread.start()

                processed_frames = 0
                with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                    logger.info(f"Device: {self.device.upper()} | ThreadPool workers: {self._max_workers}")
                    future = None
                    while True:
                        batch = read_queue.get()
                        if batch is stop_signal:
                            if future:
                                results = future.result()
                                write_queue.put(results)
                                processed_frames += len(results)
                            break

                        if future:
                            results = future.result()
                            write_queue.put(results)
                            processed_frames += len(results)

                        future = executor.submit(self._process_video_batch_async, batch)

                        if processed_frames % 50 == 0 and processed_frames > 0:
                            logger.info(f"Progress: {processed_frames}/{total_frames} frames processed.")

                write_queue.put(stop_signal)
                writer_thread.join()
                cap.release()
                out.release()

                ## wait for audio extraction to finish before merge
                audio_thread.join()

                logger.info(f"Finished processing {processed_frames} frames.")

                ## determine if re-encoding is needed
                reencode_needed = output_resolution != (width, height)
                if reencode_needed:
                    logger.info(
                        f"Resolution changed from {width}x{height} → {output_resolution[0]}x{output_resolution[1]}. "
                        f"Re-encoding will be used."
                    )

                logger.info("Merging audio back into the final video...")
                processor.recombine_video(
                    video_only_path=temp_video_path,
                    output_path=output_video_path,
                    reencode=reencode_needed
                )
                temp_video_path.unlink(missing_ok=True)

            logger.info(f"Video Pipeline completed successfully. Output saved to: {output_video_path}")

        except (FFmpegError, Exception) as e:
            logger.critical(f"Video Pipeline run failed: {e}", exc_info=True)

    def _process_video_batch_async(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Processes a batch of frames asynchronously."""
        try:
            batch_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().pin_memory()
            if self._executer._use_cuda:
                batch_tensor = batch_tensor.cuda(non_blocking=True)

            with torch.no_grad():
                processed_batch = self._safe_execute_batch(batch_tensor)

            final_batch_rgb = processed_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            del batch_tensor, processed_batch
            if self._executer._use_cuda:
                torch.cuda.empty_cache()

            return final_batch_rgb
        except Exception as e:
            logger.error(f"Async batch video processing failed: {e}", exc_info=True)
            return []

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
    
    def _auto_select_batch_size(self):
        base_size = 16
        max_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU VRAM: {max_vram:.2f} GB")

        if max_vram > 12:
            return 64
        elif max_vram > 8:
            return 32
        else:
            return base_size