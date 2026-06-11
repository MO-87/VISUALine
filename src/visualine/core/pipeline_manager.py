import importlib
import logging
import threading
import queue
import gc
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from visualine.core.config_loader import BaseConfigLoader
from visualine.core.node_base import NodeBase
from visualine.core.task_executer import TaskExecuter
from visualine.utils.file_io import VideoProcessor
from visualine.utils.resource_utils import get_resource_path

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


class PipelineManager:

    def __init__(self, config_loader: BaseConfigLoader, node_namespace: str = "visualine.nodes"):
        self._config_loader = config_loader
        self._node_namespace = node_namespace
        self._pipeline: List[NodeBase] = []
        self._executer = TaskExecuter()

        self._batch_size = 1
        self._loaded_pipeline_config_path: Optional[Path] = None
        self.config: Optional[Dict[str, Any]] = None

        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return bool(self._pipeline)

    @property
    def loaded_pipeline_config_path(self) -> Optional[Path]:
        return self._loaded_pipeline_config_path

    def _calibrate_batch_size(self, sample_frame: np.ndarray, max_batch: int = 256) -> int:
        if self.device != "cuda":
            return 4 if self.device == "mps" else 2
        
        vram_thrshld = 0.85
        free_mem, total_mem = torch.cuda.mem_get_info()
        available_vram = free_mem * vram_thrshld
        
        logger.info(f"Calibrating optimal batch size (Target limit: {vram_thrshld * 100}% of {total_mem / (1024**3):.1f} GB)...")

        def test_batch_memory(b_size: int) -> int:
            """Runs a dummy forward pass and returns peak memory usage."""
            torch.cuda.reset_peak_memory_stats()
            # Handle 4D or 5D inputs
            if sample_frame.ndim == 3:
                dummy_batch = np.repeat(np.expand_dims(sample_frame, axis=0), b_size, axis=0)
            else:
                dummy_batch = np.repeat(np.expand_dims(sample_frame, axis=0), b_size, axis=0)
            
            try:
                with torch.no_grad():
                    tensor_dummy = self._prepare_tensor_input(dummy_batch)
                    _ = self._executer(tensor_dummy)
                
                return torch.cuda.max_memory_allocated()
            except torch.cuda.OutOfMemoryError:
                return -1
            finally:
                del dummy_batch
                if 'tensor_dummy' in locals():
                    del tensor_dummy
                torch.cuda.empty_cache()

        mem_bs1 = test_batch_memory(1)
        if mem_bs1 == -1 or mem_bs1 > available_vram:
            logger.warning("VRAM limit exceeded at batch size 1. Defaulting to 1.")
            return 1

        mem_bs2 = test_batch_memory(2)
        if mem_bs2 == -1 or mem_bs2 > available_vram:
            logger.info("VRAM limit exceeded at batch size 2. Settling on 1.")
            return 1

        mem_per_sample = mem_bs2 - mem_bs1
        overhead = mem_bs1 - mem_per_sample

        if mem_per_sample <= 0:
            logger.warning("Memory scaling is non-linear or cached. Falling back to batch size 2.")
            return 2

        predicted_batch = int((available_vram - overhead) / mem_per_sample)
        predicted_batch = max(1, min(predicted_batch, max_batch))

        optimal_batch = 1
        while optimal_batch * 2 <= predicted_batch:
            optimal_batch *= 2

        logger.info(f"Math prediction: {predicted_batch}. Snapping to power of 2: {optimal_batch}.")

        if optimal_batch > 2:
            verify_mem = test_batch_memory(optimal_batch)
            if verify_mem == -1 or verify_mem > available_vram:
                logger.warning(f"Verification failed for {optimal_batch}. Falling back to {optimal_batch // 2}.")
                return optimal_batch // 2
            
            logger.info(f"Calibration successful! Settled on {optimal_batch} using {verify_mem / (1024**3):.2f} GB.")
        
        return optimal_batch

    def load_pipeline_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Loads a pipeline from a dictionary instead of a file."""
        logger.info(f"Loading pipeline from dictionary: {config_dict.get('pipeline_name', 'Unnamed')}")
        self.config = config_dict
        self._initialize_pipeline(config_dict)

    def load_pipeline(self, pipeline_config_path: Path, force_reload: bool = True) -> None:
        pipeline_config_path = Path(pipeline_config_path)
        resolved_config_path = get_resource_path(pipeline_config_path)

        if (
            not force_reload
            and self._pipeline
            and self._loaded_pipeline_config_path is not None
            and self._loaded_pipeline_config_path == pipeline_config_path
        ):
            logger.info(f"Pipeline already loaded from {pipeline_config_path}; skipping reload.")
            return

        logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")

        if self._pipeline:
            logger.info("Tearing down the existing pipeline before loading a new one...")
            self.teardown(clear_pipeline=True)

        config = self._config_loader.load(resolved_config_path)
        self.config = config
        self._initialize_pipeline(config)
        self._loaded_pipeline_config_path = pipeline_config_path

    def _initialize_pipeline(self, config: Dict[str, Any]) -> None:
        node_configs = config.get("pipeline", [])

        if not node_configs:
            raise ValueError("Pipeline configuration is empty.")

        new_pipeline: List[NodeBase] = []

        try:
            new_pipeline = [self._create_node_instance(cfg) for cfg in node_configs]
            logger.info(f"Loaded {len(new_pipeline)} nodes successfully.")

            logger.info("Setting up pipeline nodes...")
            device_obj = torch.device(self.device)

            for node in new_pipeline:
                try:
                    node.setup(device=device_obj)
                except Exception as e:
                    logger.error(f"Failed to setup node {node.node_name}: {e}", exc_info=True)
                    raise

            self._pipeline = new_pipeline
            self._executer.compile(self._pipeline)
            logger.info("All nodes set up and execution graph compiled successfully.")

        except Exception:
            logger.error("Pipeline setup failed. Cleaning up partially initialized nodes...", exc_info=True)
            for node in new_pipeline:
                try:
                    node.teardown()
                except Exception:
                    pass
            self._pipeline = []
            self._empty_device_cache()
            raise

    def teardown(self, clear_pipeline: bool = True) -> None:
        if not self._pipeline:
            logger.info("No pipeline nodes to teardown.")
            self._empty_device_cache()
            return

        logger.info("Tearing down pipeline nodes...")
        for node in self._pipeline:
            try:
                node.teardown()
            except Exception:
                logger.warning(f"Failed to teardown node: {getattr(node, 'node_name', type(node).__name__)}", exc_info=True)

        self._empty_device_cache()

        if clear_pipeline:
            self._pipeline = []
            self._loaded_pipeline_config_path = None
            self.config = None

        logger.info("Pipeline teardown complete.")

    def run(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[..., None]] = None,
        teardown_after_run: bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        if not self._pipeline:
            raise RuntimeError("No pipeline loaded — cannot run.")

        self._reset_node_runtime_state()

        try:
            # Check if any node has an explicit batch_size of 1 (common for TRT)
            if self.config and "pipeline" in self.config:
                for node_cfg in self.config["pipeline"]:
                    if node_cfg.get("params", {}).get("batch_size") == 1:
                        self._batch_size = 1
                        logger.info("Batch size forced to 1 by configuration.")
                        break

            logger.info(f"Initialized PipelineManager on {self.device.upper()} | batch_size={self._batch_size}")
            input_path = Path(input_path)
            output_path = Path(output_path)
            file_suffix = input_path.suffix.lower()

            if input_path.is_dir():
                logger.info(f"Detected image directory input: {input_path}")
                self._run_image_batch(input_path, output_path, progress_callback)
            elif file_suffix in SUPPORTED_IMAGE_EXTENSIONS:
                logger.info(f"Detected single image input: {input_path}")
                self._run_image(input_path, output_path, progress_callback)
            elif file_suffix in SUPPORTED_VIDEO_EXTENSIONS:
                logger.info(f"Detected video input: {input_path}")
                self._run_video(input_path, output_path, progress_callback, max_frames=max_frames)
            else:
                raise ValueError(f"Unsupported file type: '{file_suffix}'.")
        finally:
            if teardown_after_run:
                self.teardown(clear_pipeline=True)

    def _empty_device_cache(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        gc.collect()

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[..., None]],
        current: int,
        total: int,
        status: Optional[str] = None,
    ) -> None:
        if not progress_callback:
            return
        try:
            if status is not None:
                progress_callback(current, total, status)
            else:
                progress_callback(current, total)
        except TypeError:
            progress_callback(current, total)

    def _reset_node_runtime_state(self) -> None:
        for node in self._pipeline:
            reset_fn = getattr(node, "reset_state", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    logger.warning(f"Failed to reset runtime state for node: {node.node_name}", exc_info=True)

    def _prepare_tensor_input(self, batch_array: np.ndarray) -> torch.Tensor:
        if batch_array.ndim == 4:
            batch_array = batch_array.transpose(0, 3, 1, 2)
        elif batch_array.ndim == 5:
            batch_array = batch_array.transpose(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unsupported input batch shape: {batch_array.shape}")

        tensor = torch.from_numpy(batch_array)
        if self.device == "cuda":
            tensor = tensor.pin_memory()
        
        tensor = tensor.to(self.device, non_blocking=(self.device == "cuda"))

        if self.device == "cuda" and tensor.ndim == 4:
            return tensor.to(dtype=torch.float32, memory_format=torch.channels_last)

        return tensor.to(dtype=torch.float32)

    def _ensure_numpy_output(self, data: Union[torch.Tensor, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        data = self._unwrap_pipeline_output(data)

        if isinstance(data, torch.Tensor):
            data = data.detach().clamp(0, 255)
            if data.dtype != torch.uint8:
                data = data.to(torch.uint8)

            if data.ndim == 4:
                data = data.permute(0, 2, 3, 1).contiguous()
            elif data.ndim == 5:
                data = data.permute(0, 1, 3, 4, 2).contiguous()
            else:
                raise ValueError(f"Unsupported tensor output shape: {tuple(data.shape)}")

            return data.cpu().numpy()

        if isinstance(data, np.ndarray):
            return np.clip(data, 0, 255).astype(np.uint8, copy=False)

        raise TypeError(f"Expected torch.Tensor or np.ndarray output, got: {type(data)}")

    def _safe_execute_batch(self, batch_array: np.ndarray) -> Union[torch.Tensor, np.ndarray, Dict[str, Any]]:
        try:
            tensor_input = self._prepare_tensor_input(batch_array)
            return self._executer(tensor_input)
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
            error_msg = str(e).lower()
            is_oom = "out of memory" in error_msg or "oom" in error_msg or isinstance(e, MemoryError)
            if not is_oom:
                raise

            self._empty_device_cache()
            batch_len = len(batch_array)
            if batch_len > 1:
                logger.warning(f"OOM: reducing batch size from {batch_len} → {batch_len // 2}")
                mid = batch_len // 2
                part1 = self._safe_execute_batch(batch_array[:mid])
                part1 = self._unwrap_pipeline_output(part1)
                if isinstance(part1, torch.Tensor):
                    part1 = part1.cpu()
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                self._empty_device_cache()

                part2 = self._safe_execute_batch(batch_array[mid:])
                part2 = self._unwrap_pipeline_output(part2)
                if isinstance(part2, torch.Tensor):
                    part2 = part2.cpu()

                if isinstance(part1, torch.Tensor) and isinstance(part2, torch.Tensor):
                    return torch.cat([part1, part2], dim=0)
                return np.concatenate([np.asarray(part1), np.asarray(part2)], axis=0)

            logger.critical("OOM on a single frame. Cannot reduce batch size further.")
            raise

    def _get_output_video_spec(
        self,
        processed_sample: Union[torch.Tensor, np.ndarray, Dict[str, Any]],
    ) -> tuple[int, int, int]:
        processed_sample = self._unwrap_pipeline_output(processed_sample)

        if isinstance(processed_sample, torch.Tensor):
            shape = tuple(processed_sample.shape)
            if len(shape) == 4:
                _, output_channels, output_height, output_width = shape
            elif len(shape) == 5:
                _, _, output_channels, output_height, output_width = shape
            else:
                raise ValueError(f"Unsupported sample tensor output shape: {shape}")
            return output_width, output_height, output_channels

        if isinstance(processed_sample, np.ndarray):
            shape = processed_sample.shape
            if len(shape) == 4:
                _, output_height, output_width, output_channels = shape
            elif len(shape) == 5:
                _, _, output_height, output_width, output_channels = shape
            else:
                raise ValueError(f"Unsupported sample ndarray output shape: {shape}")
            return output_width, output_height, output_channels

        raise TypeError(f"Unsupported processed sample type: {type(processed_sample)}")

    def _run_image(
        self,
        input_image_path: Path,
        output_image_path: Path,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        try:
            image_bgr = cv2.imread(str(input_image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot load image: {input_image_path}")

            image_rgb = image_bgr[..., ::-1].copy()
            batch_array = np.expand_dims(image_rgb, axis=0)

            with torch.no_grad():
                processed_batch = self._safe_execute_batch(batch_array)

            final_batch_rgb = self._ensure_numpy_output(processed_batch)
            final_image_bgr = cv2.cvtColor(final_batch_rgb[0], cv2.COLOR_RGB2BGR)

            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_image_path), final_image_bgr)
            if not success:
                raise RuntimeError(f"Failed to write image to: {output_image_path}")

            self._emit_progress(progress_callback, 1, 1, "Image complete")
            logger.info(f"Image pipeline completed successfully. Output saved to: {output_image_path}")
        except Exception as e:
            logger.critical(f"Image pipeline run failed: {e}", exc_info=True)
            raise

    def _run_image_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        try:
            image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
            total_images = len(image_paths)
            if not total_images:
                raise FileNotFoundError(f"No supported images found in directory: {input_dir}")

            images_by_size: Dict[tuple, list] = {}
            skipped_images = 0

            for img_path in image_paths:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    skipped_images += 1
                    logger.warning(f"Skipping unreadable image: {img_path}")
                    continue
                h, w, _ = img_bgr.shape
                resolution = (w, h)
                if resolution not in images_by_size:
                    images_by_size[resolution] = []
                img_rgb = img_bgr[..., ::-1].copy()
                images_by_size[resolution].append((img_rgb, img_path.name))

            if not images_by_size:
                raise RuntimeError(f"No readable supported images found in directory: {input_dir}")

            output_dir.mkdir(parents=True, exist_ok=True)
            group_count = len(images_by_size)
            processed_count = 0
            readable_total = total_images - skipped_images

            with tqdm(total=readable_total, desc="Processing Images", unit="img", ncols=100) as pbar:
                for i, (resolution, image_list) in enumerate(images_by_size.items(), 1):
                    pbar.set_description(f"Group {i}/{group_count} ({resolution[0]}x{resolution[1]})")
                    
                    # Calibrate for each new resolution
                    sample_frame = image_list[0][0]
                    if self._batch_size != 1:
                        self._batch_size = self._calibrate_batch_size(sample_frame)

                    batch_buffer, batch_names = [], []
                    for img_rgb, name in image_list:
                        batch_buffer.append(img_rgb)
                        batch_names.append(name)

                        if len(batch_buffer) >= self._batch_size:
                            self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                            processed_count += len(batch_buffer)
                            pbar.update(len(batch_buffer))
                            self._emit_progress(progress_callback, processed_count, readable_total, "Processing images")
                            batch_buffer, batch_names = [], []

                    if batch_buffer:
                        self._process_and_save_image_batch(batch_buffer, batch_names, output_dir)
                        processed_count += len(batch_buffer)
                        pbar.update(len(batch_buffer))
                        self._emit_progress(progress_callback, processed_count, readable_total, "Processing images")

            logger.info(f"Image directory pipeline completed. Outputs saved to: {output_dir}")
        except Exception as e:
            logger.critical(f"Image batch pipeline run failed: {e}", exc_info=True)
            raise

    def _process_and_save_image_batch(self, images: List[np.ndarray], names: List[str], output_dir: Path) -> None:
        try:
            batch_array = np.stack(images)
            with torch.no_grad():
                processed_batch = self._safe_execute_batch(batch_array)
            final_batch_rgb = self._ensure_numpy_output(processed_batch)
            for name, img_rgb in zip(names, final_batch_rgb):
                out_path = output_dir / name
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(out_path), img_bgr)
                if not success:
                    raise RuntimeError(f"Failed to write image to: {out_path}")
        except Exception as e:
            logger.error(f"Batch image processing failed: {e}", exc_info=True)
            raise

    def _run_video(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Optional[Callable[..., None]] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        reader_process = None
        writer_process = None
        reader_thread = None
        writer_thread = None

        try:
            processor = VideoProcessor()
            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            max_ai_resolution = 1920
            reader_process, input_width, input_height, input_fps, total_frames, input_fps_str = processor.get_ffmpeg_reader(
                input_video_path, max_dimension=max_ai_resolution
            )

            if max_frames is not None:
                total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames
                logger.info(f"Limiting video processing to {total_frames} frames.")

            output_fps = input_fps
            output_fps_str = input_fps_str
            temporal_window = 1
            for node in self._pipeline:
                if hasattr(node, "fps_multiplier"):
                    output_fps *= node.fps_multiplier
                    output_fps_str = output_fps
                if hasattr(node, "temporal_window") and node.temporal_window > 1:
                    temporal_window = max(temporal_window, node.temporal_window)

            pad_frames = temporal_window // 2 if temporal_window > 1 else 0
            frame_size = input_width * input_height * 3

            raw_bytes = reader_process.stdout.read(frame_size)
            if not raw_bytes:
                raise RuntimeError("Failed to read the first frame from FFmpeg pipe.")

            first_frame_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((input_height, input_width, 3))
            
            if temporal_window > 1:
                sample_frame = np.stack([first_frame_rgb] * temporal_window)
            else:
                sample_frame = first_frame_rgb

            if self._batch_size != 1:
                self._batch_size = self._calibrate_batch_size(sample_frame)
            
            sample_array = np.expand_dims(sample_frame, axis=0)

            with torch.no_grad():
                processed_sample = self._safe_execute_batch(sample_array)
                output_width, output_height, output_channels = self._get_output_video_spec(processed_sample)

            del sample_array, processed_sample
            self._reset_node_runtime_state()

            read_queue = queue.Queue(maxsize=4)
            write_queue = queue.Queue(maxsize=4)
            error_queue = queue.Queue()
            stop_signal = object()

            temp_output_path = output_video_path.with_name(f"{output_video_path.stem}_temp{output_video_path.suffix}")
            writer_process, actual_temp_path = processor.get_ffmpeg_writer(
                temp_output_path, output_width, output_height, output_fps_str, channels=output_channels,
            )

            actual_final_path = output_video_path
            if actual_temp_path.suffix.lower() != temp_output_path.suffix.lower():
                actual_final_path = output_video_path.with_suffix(actual_temp_path.suffix)

            def frame_reader():
                try:
                    buffer = deque(maxlen=temporal_window) if temporal_window > 1 else None
                    batch_buffer = []

                    frame_rgb = first_frame_rgb
                    if temporal_window > 1:
                        for _ in range(pad_frames):
                            buffer.append(frame_rgb)
                        buffer.append(frame_rgb)
                        for _ in range(pad_frames):
                            raw_next = reader_process.stdout.read(frame_size)
                            if raw_next:
                                buffer.append(np.frombuffer(raw_next, dtype=np.uint8).reshape((input_height, input_width, 3)))
                            else:
                                buffer.append(buffer[-1])
                    else:
                        batch_buffer.append(frame_rgb)

                    frames_read = 1 # We already read the first frame
                    while True:
                        if max_frames is not None and frames_read >= max_frames:
                            break

                        if temporal_window > 1:
                            batch_buffer.append(np.stack(list(buffer)))
                        if len(batch_buffer) >= self._batch_size:
                            read_queue.put(np.stack(batch_buffer))
                            batch_buffer = []

                        raw_frame_bytes = reader_process.stdout.read(frame_size)
                        if not raw_frame_bytes:
                            break
                        
                        frames_read += 1
                        frame_rgb = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape((input_height, input_width, 3))
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
                except Exception as e:
                    error_queue.put(e)
                finally:
                    read_queue.put(stop_signal)
                    try: reader_process.stdout.close()
                    except Exception: pass

            def frame_writer():
                try:
                    while True:
                        batch = write_queue.get()
                        if batch is stop_signal:
                            break
                        if writer_process.poll() is not None:
                            raise RuntimeError(f"FFmpeg writer exited prematurely with code {writer_process.returncode}")
                        for frame in batch:
                            writer_process.stdin.write(frame.tobytes())
                        writer_process.stdin.flush()
                except Exception as e:
                    error_queue.put(e)
                finally:
                    try: writer_process.stdin.close()
                    except Exception: pass

            reader_thread = threading.Thread(target=frame_reader, daemon=True)
            writer_thread = threading.Thread(target=frame_writer, daemon=True)
            reader_thread.start()
            writer_thread.start()

            with tqdm(total=total_frames, desc="Processing Video", unit="fr", ncols=100) as pbar:
                while True:
                    if not error_queue.empty():
                        raise error_queue.get()
                    batch_array = read_queue.get()
                    if batch_array is stop_signal:
                        write_queue.put(stop_signal)
                        break
                    with torch.no_grad():
                        processed_batch = self._safe_execute_batch(batch_array)
                    final_batch = self._ensure_numpy_output(processed_batch)
                    write_queue.put(final_batch)
                    pbar.update(len(batch_array))
                    self._emit_progress(progress_callback, pbar.n, total_frames if total_frames > 0 else pbar.n, "Processing video")
                    del batch_array, processed_batch, final_batch

            reader_thread.join()
            writer_thread.join()

            # Ensure the writer process has completely finished and flushed all data (moov atom)
            if writer_process:
                writer_process.wait()

            if not error_queue.empty():
                raise error_queue.get()

            if actual_temp_path.exists():
                processor.mux_audio(input_video_path, actual_temp_path, actual_final_path)
                actual_temp_path.unlink()
            logger.info(f"Video Pipeline completed successfully. Output saved to {actual_final_path}")
        except Exception as e:
            logger.critical(f"Video Pipeline run failed: {e}", exc_info=True)
            raise
        finally:
            for proc in (reader_process, writer_process):
                if proc:
                    try:
                        if proc.poll() is None: proc.terminate()
                    except Exception: pass

    def _create_node_instance(self, node_config: Dict[str, Any]) -> NodeBase:
        class_path = node_config.get("class")
        if not class_path:
            raise ValueError("Node configuration missing 'class' key.")
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            full_module_path = f"{self._node_namespace}.{module_path}"
            node_module = importlib.import_module(full_module_path)
            node_class = getattr(node_module, class_name)
            return node_class(config=node_config.get("params", {}))
        except Exception as e:
            raise ImportError(f"Could not load class '{class_path}': {e}") from e
    
    def _unwrap_pipeline_output(self, data: Any) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(data, dict):
            if "tensor" not in data:
                raise TypeError("Pipeline output is a dict but does not contain a 'tensor' key.")
            return data["tensor"]
        return data
