import importlib
import logging
import threading
import queue
import gc
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Union, Callable, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from visualine.core.config_loader import BaseConfigLoader, get_resource_path
from visualine.core.node_base import NodeBase
from visualine.core.task_executer import TaskExecuter
from visualine.utils.file_io import VideoProcessor

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class PipelineManager:

    def __init__(self, config_loader: BaseConfigLoader, node_namespace: str = "visualine.nodes"):
        self._config_loader = config_loader
        self._node_namespace = node_namespace
        self._pipeline: List[NodeBase] = []
        self._executer = TaskExecuter()
        
        self._batch_size = 1
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            

    def load_pipeline(self, pipeline_config_path: Path) -> None:
        logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")
        config = self._config_loader.load(get_resource_path(pipeline_config_path))
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
        
        logger.info("Setting up pipeline nodes...")
        device_obj = torch.device(self.device)
        for node in self._pipeline:
            try:
                node.setup(device=device_obj)
            except Exception as e:
                logger.error(f"Failed to setup node {node.node_name}: {e}", exc_info=True)
                raise
                
        self._executer.compile(self._pipeline)
        logger.info("All nodes set up and execution graph compiled successfully.")

    def run(self, input_path: Path, output_path: Path, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> None:
        if not self._pipeline:
            logger.error("No pipeline loaded — cannot run.")
            return

        try:
            logger.info(f"Initialized PipelineManager on {self.device.upper()} | batch_size={self._batch_size}")
            file_suffix = input_path.suffix.lower()

            if input_path.is_dir():
                logger.info(f"Detected image directory input: {input_path}")
                self._run_image_batch(input_path, output_path, progress_callback)
            elif file_suffix in SUPPORTED_IMAGE_EXTENSIONS:
                logger.info(f"Detected single image input: {input_path}")
                self._run_image(input_path, output_path, progress_callback)
            elif file_suffix in SUPPORTED_VIDEO_EXTENSIONS:
                logger.info(f"Detected video input: {input_path}")
                self._run_video(input_path, output_path, progress_callback)
            else:
                raise ValueError(f"Unsupported file type: '{file_suffix}'.")
        finally:
            logger.info("Tearing down pipeline nodes...")
            for node in self._pipeline:
                node.teardown()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Pipeline teardown complete.")

    def _prepare_tensor_input(self, batch_array: np.ndarray) -> torch.Tensor:
        if batch_array.ndim == 4:
            ## (B, H, W, C) -> (B, C, H, W)
            batch_array = batch_array.transpose(0, 3, 1, 2)
        elif batch_array.ndim == 5:
            ## (B, T, H, W, C) -> (B, T, C, H, W)
            batch_array = batch_array.transpose(0, 1, 4, 2, 3)
            
        tensor = torch.from_numpy(batch_array)
        
        if self.device in ["cuda", "mps"]:
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32  ## CPU requires float32 for most operations
            
        if self.device == "cuda":
            tensor = tensor.to(self.device, non_blocking=True)
        else:
            tensor = tensor.to(self.device)
            
        return tensor.to(dtype=target_dtype, memory_format=torch.contiguous_format)

    def _ensure_numpy_output(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            if data.dtype != torch.uint8:
                data = data.to(torch.uint8)
            return data.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return data

    def _safe_execute_batch(self, batch_array: np.ndarray) -> torch.Tensor:
        try:
            tensor_input = self._prepare_tensor_input(batch_array)
            return self._executer(tensor_input)
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg or isinstance(e, MemoryError):
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()

                batch_len = len(batch_array)
                if batch_len > 1:
                    logger.warning(f"Memory OOM. Splitting batch of {batch_len} in half and retrying...")
                    mid = batch_len // 2
                    part1 = self._safe_execute_batch(batch_array[:mid])
                    part2 = self._safe_execute_batch(batch_array[mid:])
                    
                    return torch.cat([part1, part2])
                else:
                    logger.critical("OOM on a single frame. Cannot reduce batch size further.")
                    raise
            else:
                raise

    def _run_image(self, input_image_path: Path, output_image_path: Path, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        try:
            image_bgr = cv2.imread(str(input_image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot load image: {input_image_path}")
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            batch_array = np.expand_dims(image_rgb, axis=0)

            with torch.no_grad():
                processed_batch = self._safe_execute_batch(batch_array)
            
            final_batch_rgb = self._ensure_numpy_output(processed_batch)
            final_image_bgr = cv2.cvtColor(final_batch_rgb[0], cv2.COLOR_RGB2BGR)
            
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), final_image_bgr)

            if progress_callback:
                progress_callback(1, 1)

            logger.info(f"Image pipeline completed successfully. Output saved to: {output_image_path}")
        except Exception as e:
            logger.critical(f"Image pipeline run failed: {e}", exc_info=True)

    def _run_image_batch(self, input_dir: Path, output_dir: Path, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        try:
            image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
            total_images = len(image_paths)
            if not total_images:
                raise FileNotFoundError(f"No supported images found in directory: {input_dir}")

            images_by_size: Dict[tuple, list] = {}
            for img_path in image_paths:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                h, w, _ = img_bgr.shape
                resolution = (w, h)
                if resolution not in images_by_size:
                    images_by_size[resolution] = []
                    
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                images_by_size[resolution].append((img_rgb, img_path.name))

            output_dir.mkdir(parents=True, exist_ok=True)
            group_count = len(images_by_size)

            with tqdm(total=total_images, desc="Processing Images", unit="img", ncols=100) as pbar:
                for i, (resolution, image_list) in enumerate(images_by_size.items(), 1):
                    pbar.set_description(f"Group {i}/{group_count} ({resolution[0]}x{resolution[1]})")
                    
                    sample_frame = image_list[0][0]
                    
                    batch_buffer, batch_names = [], []
                    for img_rgb, name in image_list:
                        batch_buffer.append(img_rgb)
                        batch_names.append(name)

                        if len(batch_buffer) >= self._batch_size:
                            self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                            pbar.update(len(batch_buffer))
                            if progress_callback:
                                progress_callback(pbar.n, total_images)
                            batch_buffer, batch_names = [], []
                    
                    if batch_buffer:
                        self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                        pbar.update(len(batch_buffer))
                        if progress_callback:
                            progress_callback(pbar.n, total_images)

            logger.info(f"Image directory pipeline completed. Outputs saved to: {output_dir}")
        except Exception as e:
            logger.critical(f"Image batch pipeline run failed: {e}", exc_info=True)

    def _process_and_save_image_batch(self, images: List[np.ndarray], names: List[str], output_dir: Path) -> None:
        try:
            batch_array = np.stack(images)
            with torch.no_grad():
                processed_batch = self._safe_execute_batch(batch_array)
            
            final_batch_rgb = self._ensure_numpy_output(processed_batch)

            for name, img_rgb in zip(names, final_batch_rgb):
                out_path = output_dir / name
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), img_bgr)

        except Exception as e:
            logger.error(f"Batch image processing failed: {e}", exc_info=True)

    def _run_video(self, input_video_path: Path, output_video_path: Path, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        try:
            cap = cv2.VideoCapture(str(input_video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {input_video_path}")

            processor = VideoProcessor()
            input_fps = processor.get_framerate(input_video_path)
            
            output_fps = input_fps
            temporal_window = 1

            for node in self._pipeline:
                if hasattr(node, 'fps_multiplier'):
                    output_fps *= node.fps_multiplier
                if hasattr(node, 'temporal_window') and node.temporal_window > 1:
                    temporal_window = max(temporal_window, node.temporal_window)

            pad_frames = temporal_window // 2 if temporal_window > 1 else 0

            input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ret, first_frame_bgr = cap.read()
            first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
            
            if temporal_window > 1:
                sample_frame = np.stack([first_frame_rgb] * temporal_window)
            else:
                sample_frame = first_frame_rgb

            
            sample_array = np.expand_dims(sample_frame, axis=0)
            
            with torch.no_grad():
                tensor_sample = self._prepare_tensor_input(sample_array)
                processed_sample = self._executer(tensor_sample)
                if isinstance(processed_sample, torch.Tensor):
                    _, _, output_height, output_width = processed_sample.shape
                else:
                    _, output_height, output_width, _ = processed_sample.shape
            
            del sample_array, processed_sample
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            read_queue = queue.Queue(maxsize=2)
            write_queue = queue.Queue(maxsize=2)
            stop_signal = object()

            def frame_reader():
                buffer = deque(maxlen=temporal_window) if temporal_window > 1 else None
                batch_buffer = []

                ret, frame_bgr = cap.read()
                if not ret:
                    read_queue.put(stop_signal)
                    return
                
                frame_rgb = frame_bgr[..., ::-1]

                if temporal_window > 1:
                    for _ in range(pad_frames):
                        buffer.append(frame_rgb)
                    buffer.append(frame_rgb)
                    
                    for _ in range(pad_frames):
                        ret, next_bgr = cap.read()
                        if ret:
                            buffer.append(cv2.cvtColor(next_bgr, cv2.COLOR_BGR2RGB))
                        else:
                            buffer.append(buffer[-1])
                else:
                    batch_buffer.append(frame_rgb)

                while True:
                    if temporal_window > 1:
                        batch_buffer.append(np.stack(list(buffer)))

                    if len(batch_buffer) >= self._batch_size:
                        read_queue.put(np.stack(batch_buffer))
                        batch_buffer = []

                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    if temporal_window > 1:
                        buffer.append(frame_rgb)
                    else:
                        batch_buffer.append(frame_rgb)

                if temporal_window > 1:
                    for _ in range(pad_frames):
                        buffer.append(buffer[-1])
                        batch_buffer.append(np.stack(list(buffer)))
                        if len(batch_buffer) >= self._batch_size:
                            read_queue.put(np.stack(batch_buffer))
                            batch_buffer = []

                if batch_buffer:
                    read_queue.put(np.stack(batch_buffer))

                read_queue.put(stop_signal)

            def frame_writer():
                process = processor.get_ffmpeg_writer(
                    input_video_path, output_video_path, output_width, output_height, output_fps
                )
                try:
                    while True:
                        batch = write_queue.get()
                        if batch is stop_signal:
                            break
                        process.stdin.write(batch.tobytes())
                        write_queue.task_done()
                finally:
                    process.stdin.close()
                    process.wait()

            reader_thread = threading.Thread(target=frame_reader, daemon=True)
            writer_thread = threading.Thread(target=frame_writer, daemon=True)
            
            reader_thread.start()
            writer_thread.start()

            with tqdm(total=total_frames, desc="Processing Video", unit="fr", ncols=100) as pbar:
                while True:
                    batch_array = read_queue.get()
                    if batch_array is stop_signal:
                        write_queue.put(stop_signal)
                        break

                    with torch.no_grad():
                        processed_batch = self._safe_execute_batch(batch_array)

                    final_batch = self._ensure_numpy_output(processed_batch)
                    
                    if final_batch.shape[-1] == 1:
                        final_batch = np.repeat(final_batch, 3, axis=-1)

                    write_queue.put(final_batch)
                    pbar.update(len(batch_array))

                    del batch_array, processed_batch

                    if progress_callback:
                        progress_callback(pbar.n, total_frames)

            writer_thread.join()
            cap.release()
            logger.info("Video Pipeline completed successfully.")

        except Exception as e:
            logger.critical(f"Video Pipeline run failed: {e}", exc_info=True)

    def _create_node_instance(self, node_config: Dict[str, Any]) -> NodeBase:
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