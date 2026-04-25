import logging
import shutil
import subprocess
import json
from pathlib import Path
import threading
import sys

logger = logging.getLogger(__name__)

class FFmpegError(Exception):
    pass

class VideoProcessor:

    def __init__(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise FFmpegError("FFmpeg/ffprobe is not installed or not in system's PATH.")
        
        self._available_encoders = self._get_available_encoders()

    def _get_available_encoders(self) -> str:
        try:
            res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
            return res.stdout
        except Exception as e:
            logger.warning(f"Failed to query FFmpeg encoders: {e}")
            return ""

    def _get_encoder_flags(self, width: int, height: int) -> list:
        """Dynamically detects best encoder and switches to HEVC (H.265) for 8K support."""
        
        needs_hevc = width > 4096 or height > 2304
        encoders = self._available_encoders

        if needs_hevc:
            logger.info(f"Resolution {width}x{height} exceeds H.264 limits. Switching to HEVC (H.265).")
            if "hevc_nvenc" in encoders:
                return ["-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "19"]
            elif "hevc_videotoolbox" in encoders and sys.platform == "darwin":
                return ["-c:v", "hevc_videotoolbox", "-q:v", "50"]
            elif "hevc_amf" in encoders:
                return ["-c:v", "hevc_amf", "-quality", "quality"]
            elif "hevc_qsv" in encoders:
                return ["-c:v", "hevc_qsv", "-global_quality", "19"]
            else:
                return ["-c:v", "libx265", "-preset", "fast", "-crf", "19"]
        else:
            if "h264_nvenc" in encoders:
                return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"]
            elif "h264_videotoolbox" in encoders and sys.platform == "darwin":
                return ["-c:v", "h264_videotoolbox", "-q:v", "50"]
            elif "h264_amf" in encoders:
                return ["-c:v", "h264_amf", "-quality", "quality"]
            elif "h264_qsv" in encoders:
                return ["-c:v", "h264_qsv", "-global_quality", "19"]
            else:
                return ["-c:v", "libx264", "-preset", "fast", "-crf", "19"]

    def get_video_info(self, video_path: Path) -> dict:
        """Safely extracts comprehensive metadata using ffprobe JSON output."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
            if not video_stream:
                raise ValueError("No video stream found.")

            fps_str = video_stream.get("r_frame_rate", "30/1")
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom if denom != 0 else 30.0

            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": fps,
                "frames": int(video_stream.get("nb_frames", 0))
            }
        except Exception as e:
            logger.error(f"Failed to extract video info for {video_path}: {e}")
            raise

    def get_ffmpeg_reader(self, input_video_path: Path, max_dimension: int = 1920):
        """
        Creates a reading pipe that handles high-quality downscaling natively in FFmpeg.
        This saves massive amounts of Python/PyTorch memory for 4K+ inputs.
        """
        info = self.get_video_info(input_video_path)
        orig_w, orig_h = info["width"], info["height"]
        
        if max(orig_w, orig_h) > max_dimension:
            if orig_w > orig_h:
                new_w = max_dimension
                new_h = int((max_dimension / orig_w) * orig_h)
            else:
                new_h = max_dimension
                new_w = int((max_dimension / orig_h) * orig_w)
            
            new_w, new_h = new_w - (new_w % 2), new_h - (new_h % 2)
            logger.info(f"Downscaling input from {orig_w}x{orig_h} to {new_w}x{new_h} for AI processing.")
        else:
            new_w, new_h = orig_w, orig_h

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "auto",
            "-i", str(input_video_path),
            "-vf", f"scale={new_w}:{new_h}:flags=lanczos",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", "-"
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
        
        return process, new_w, new_h, info["fps"], info.get("frames", 0)

    def get_ffmpeg_writer(self, output_video_path: Path, width: int, height: int, fps: float):
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        encoder_flags = self._get_encoder_flags(width, height)
        
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo", 
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", 
            *encoder_flags, 
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", 
            str(output_video_path)
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        def log_stderr(pipe):
            for line in iter(pipe.readline, b''):
                logger.warning(f"FFmpeg Writer: {line.decode('utf-8').strip()}")
        
        threading.Thread(target=log_stderr, args=(process.stderr,), daemon=True).start()
        
        return process

    def mux_audio(self, original_video_path: Path, silent_video_path: Path, final_output_path: Path):
        """Instantly copies the audio from the original video into the new processed video."""
        logger.info("Muxing original audio back into the upscaled video...")
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent_video_path),
            "-i", str(original_video_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c", "copy",
            "-shortest",
            str(final_output_path)
        ]
        subprocess.run(command, check=True)