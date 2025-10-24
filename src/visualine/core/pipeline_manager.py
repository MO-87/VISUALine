import os
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import cv2
import numpy as np
import torch
import av

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
                tensor = tensor.pin_memory().cuda(non_blocking=True)

            processed_tensor = self._executer.execute_batch(self._pipeline, tensor)

            final_image_rgb = processed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            final_image = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), final_image)

            logger.info(f"Image pipeline completed successfully. Output saved to: {output_image_path}")
        except Exception as e:
            logger.critical(f"Image pipeline run failed: {e}", exc_info=True)

    def _run_image_batch(self, input_dir: Path, output_dir: Path) -> None:
        """
        Groups images by resolution and runs the pipeline on each group to preserve aspect ratios.
        """
        try:
            image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
            if not image_paths:
                raise FileNotFoundError(f"No supported images found in directory: {input_dir}")

            logger.info("Grouping images by resolution...")
            images_by_size: Dict[tuple, list] = {}
            for img_path in image_paths:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.warning(f"Skipping unreadable image: {img_path}")
                    continue
                
                h, w, _ = img_bgr.shape
                resolution = (w, h)
                
                if resolution not in images_by_size:
                    images_by_size[resolution] = []
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                images_by_size[resolution].append((img_rgb, img_path.name))

            logger.info(f"Found {len(images_by_size)} resolution groups to process.")
            output_dir.mkdir(parents=True, exist_ok=True)

            total_processed = 0
            group_count = len(images_by_size)
            for i, (resolution, image_list) in enumerate(images_by_size.items(), 1):
                logger.info(
                    f"Processing group {i}/{group_count}: "
                    f"Resolution {resolution[0]}x{resolution[1]} ({len(image_list)} images)"
                )
                
                batch_buffer, batch_names = [], []
                for img_rgb, name in image_list:
                    batch_buffer.append(img_rgb)
                    batch_names.append(name)

                    if len(batch_buffer) >= self._batch_size:
                        self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                        total_processed += len(batch_buffer)
                        batch_buffer, batch_names = [], []
                
                if batch_buffer:
                    self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                    total_processed += len(batch_buffer)

                logger.info(f"Progress: {total_processed}/{len(image_paths)} total images processed.")

            logger.info(f"Image directory pipeline completed successfully. Outputs saved to: {output_dir}")
        except Exception as e:
            logger.critical(f"Image batch pipeline run failed: {e}", exc_info=True)

    def _process_and_save_image_batch(self, images: List[np.ndarray], names: List[str], output_dir: Path) -> None:
        """Processes and saves a batch of images."""
        try:
            batch_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
            if self._executer._use_cuda:
                batch_tensor = batch_tensor.pin_memory().cuda(non_blocking=True)

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
            logger.warning(f"GPU OOM detected. Reducing batch size to {self._batch_size}.")
            raise

    def _run_video(self, input_video_path: Path, output_video_path: Path) -> None:
        """Runs the pipeline on a video file using async batched GPU-optimized processing."""
        try:
            with VideoProcessor(input_video_path) as processor:
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    raise FileNotFoundError(f"Cannot open video: {input_video_path}")

                fps = processor.get_framerate()
                input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                ## determine output resolution by processing a sample frame through the entire pipeline
                ret, first_frame_bgr = cap.read()
                if not ret:
                    raise ValueError("Cannot read first frame from video")
                
                first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
                sample_tensor = torch.from_numpy(first_frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
                
                if self._executer._use_cuda:
                    sample_tensor = sample_tensor.pin_memory().cuda(non_blocking=True)
                
                with torch.no_grad():
                    processed_sample = self._executer.execute_batch(self._pipeline, sample_tensor)
                    _, _, output_height, output_width = processed_sample.shape
                
                del sample_tensor, processed_sample
                if self._executer._use_cuda:
                    torch.cuda.empty_cache()
                
                logger.info(f"Input resolution: {input_width}x{input_height} -> Output resolution: {output_width}x{output_height}")
                
                ## reset video capture to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ## temporary encoded video path (we still write a file but encoded with PyAV)
                temp_video_path = output_video_path.with_name(f"{output_video_path.stem}_temp_video.mp4")

                ## init audio extraction in parallel
                audio_thread = threading.Thread(target=processor.extract_audio, daemon=True)
                audio_thread.start()

                read_queue = queue.Queue(maxsize=4)
                write_queue = queue.Queue(maxsize=2)
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

                ## write thread (uses PyAV now to encode frames to H.264)
                def frame_writer():
                    """
                    Encode incoming RGB frames to H.264 using PyAV and write to temp_video_path.
                    This preserves your original threading architecture while switching the encoder.
                    """
                    try:
                        container = av.open(str(temp_video_path), mode="w")
                        stream = container.add_stream("h264", rate=int(round(fps)))
                        stream.width = output_width
                        stream.height = output_height
                        stream.pix_fmt = "yuv420p"
                        ## balanced preset for speed/quality tradeoff with good compression
                        stream.options = {"preset": "medium", "crf": "23"}

                        frames_written = 0
                        while True:
                            item = write_queue.get()
                            if item is stop_signal:
                                break
                            for img_rgb in item:
                                ## img_rgb shape: (H, W, 3) in RGB
                                frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
                                ## convert/encode frame and mux packets
                                for packet in stream.encode(frame):
                                    container.mux(packet)
                                frames_written += 1
                            write_queue.task_done()

                        ## flushing the encoder
                        for packet in stream.encode():
                            container.mux(packet)
                        container.close()
                        logger.debug(f"Encoded {frames_written} frames to temporary video file.")
                    except Exception as e:
                        logger.critical(f"PyAV writer thread failed: {e}", exc_info=True)
                        raise

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

                        if processed_frames > 0 and processed_frames % 100 == 0:
                            progress_pct = (processed_frames / total_frames) * 100
                            logger.info(f"Progress: {processed_frames}/{total_frames} frames ({progress_pct:.1f}%)")

                write_queue.put(stop_signal)
                writer_thread.join()
                cap.release()

                ## wait for audio extraction to finish before merge
                audio_thread.join()

                logger.info(f"Finished processing {processed_frames} frames.")

                ## determine if re-encoding is needed (resolution changed from input to output)
                reencode_needed = (output_width, output_height) != (input_width, input_height)
                if reencode_needed:
                    logger.info(
                        f"Resolution changed from {input_width}x{input_height} to {output_width}x{output_height}. "
                        f"Re-encoding will be used."
                    )
                else:
                    logger.info(f"Resolution unchanged at {output_width}x{output_height}. Direct stream copy will be used.")

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
            batch_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
            if self._executer._use_cuda:
                batch_tensor = batch_tensor.pin_memory().cuda(non_blocking=True)

            with torch.no_grad():
                processed_batch = self._safe_execute_batch(batch_tensor)

            final_batch_any_format = processed_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            ## convert single-channel outputs to RGB if needed
            final_batch_rgb = []
            for img in final_batch_any_format:
                if img.shape[2] == 1: ## if the node returned a single-channel image
                    final_batch_rgb.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                else:
                    final_batch_rgb.append(img)

            del batch_tensor, processed_batch, final_batch_any_format
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
        """Automatically selects optimal batch size based on available GPU VRAM."""
        base_size = 4
        max_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU VRAM: {max_vram:.2f} GB")

        if max_vram >= 16:
            return 32
        elif max_vram >= 10:
            return 16
        elif max_vram >= 6:
            return 8
        else:
            return base_size