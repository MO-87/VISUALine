import atexit
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Custom exception for errors related to FFmpeg execution."""
    pass


class VideoProcessor:
    """
    A context manager for video/image I/O and processing using FFmpeg.
    """

    def __init__(self, video_path: Path = None):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise FFmpegError("FFmpeg/ffprobe is not installed or not in system's PATH.")
        if video_path and not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        self.video_path = video_path
        self.temp_dir = None
        self.frames_dir = None
        self.audio_path = None
        self._cleanup_func = None

    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="visualine_"))
        self._cleanup_func = lambda: shutil.rmtree(self.temp_dir, ignore_errors=True)
        atexit.register(self._cleanup_func)
        logger.info(f"Created temporary processing directory: {self.temp_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            if self._cleanup_func:
                atexit.unregister(self._cleanup_func)

    def __del__(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.warning(f"VideoProcessor object cleaned up dangling temp dir: {self.temp_dir}")

    def _run_command(self, command: list):
        logger.debug(f"Running FFmpeg command: {' '.join(command)}")
        try:
            subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            error_lines = (e.stderr.decode() if e.stderr else "No stderr.").splitlines()[:5]
            error_message = "\n".join(error_lines)
            logger.error(f"FFmpeg command failed: {' '.join(command)}")
            logger.error(f"FFmpeg stderr: {error_message}")
            raise FFmpegError(f"FFmpeg error: {error_message}")

    def load_image(self, image_path: Path) -> np.ndarray:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            return np.array(img.convert("RGB"))

    def save_image(self, image_array: np.ndarray, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(image_array.astype('uint8'), 'RGB')
        img.save(output_path)
        logger.info(f"Image saved to: {output_path}")

    def get_framerate(self) -> float:
        if not self.video_path: return 30.0
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", str(self.video_path)],
                capture_output=True, text=True, check=True
            )
            num, denom = map(int, result.stdout.strip().split('/'))
            return num / denom
        except Exception:
            logger.warning("Failed to detect frame rate; defaulting to 30 fps.")
            return 30.0

    def extract_frames(self, quality: int = 2, start_time: float = None, duration: float = None) -> Path:
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir()
        logger.info(f"Extracting frames from {self.video_path}...")
        frame_pattern = self.frames_dir / "frame_%06d.png"
        
        command = ["ffmpeg"]
        if start_time is not None:
            command.extend(["-ss", str(start_time)])
        command.extend(["-i", str(self.video_path)])
        if duration is not None:
            command.extend(["-t", str(duration)])
        command.extend(["-q:v", str(quality), str(frame_pattern)])
        
        self._run_command(command)
        logger.info(f"Frames extracted to: {self.frames_dir}")
        return self.frames_dir

    def extract_audio(self) -> Path | None:
        logger.info(f"Extracting audio from {self.video_path}...")
        self.audio_path = self.temp_dir / "audio.aac"
        command = ["ffmpeg", "-i", str(self.video_path), "-vn", "-acodec", "copy", str(self.audio_path)]
        
        try:
            self._run_command(command)
            if self.audio_path.exists() and self.audio_path.stat().st_size > 0:
                logger.info(f"Audio extracted to: {self.audio_path}")
                return self.audio_path
            raise FFmpegError("Audio extraction produced an empty file.")
        except FFmpegError as e:
            logger.warning(f"Audio extraction failed or no stream found: {e}")
            self.audio_path = None
            return None

    def recombine_video(self, output_path: Path, framerate: float = None, crf: int = 18):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        effective_framerate = framerate if framerate is not None else self.get_framerate()
        logger.info(f"Recombining frames into {output_path} at {effective_framerate:.2f} fps...")
        frame_pattern = self.frames_dir / "frame_%06d.png"
        command = [
            "ffmpeg", "-y", "-framerate", str(effective_framerate), "-i", str(frame_pattern),
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"
        ]
        if self.audio_path and self.audio_path.exists():
            command.extend(["-i", str(self.audio_path), "-c:a", "aac", "-shortest"])
        command.append(str(output_path))
        self._run_command(command)
        logger.info(f"Successfully created final video at: {output_path}")

    def merge_audio(self, video_input: Path, audio_input: Path, final_output: Path):
        if not audio_input or not audio_input.exists():
            logger.warning("No audio file provided to merge. Copying video file directly.")
            shutil.copy(video_input, final_output)
            return
            
        logger.info(f"Merging audio from {audio_input} into {final_output}...")
        command = [
            "ffmpeg", "-y", "-i", str(video_input), "-i", str(audio_input),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(final_output)
        ]
        self._run_command(command)
        logger.info("Successfully merged audio.")