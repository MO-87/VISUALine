import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class FFmpegError(Exception):
    pass

class VideoProcessor:

    def __init__(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise FFmpegError("FFmpeg/ffprobe is not installed or not in system's PATH.")
            
    def get_framerate(self, video_path: Path) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", str(video_path)],
                capture_output=True, text=True, check=True
            )
            num, denom = map(int, result.stdout.strip().split('/'))
            return num / denom
        except Exception:
            logger.warning("Failed to detect frame rate; defaulting to 30.0 fps.")
            return 30.0

    def get_ffmpeg_writer(self, input_video_path: Path, output_video_path: Path, 
                          width: int, height: int, fps: float):
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        nvenc_available = False
        if width <= 4096 and height <= 4096:
            try:
                res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
                nvenc_available = "h264_nvenc" in res.stdout
            except Exception:
                pass

        encoder = ["-c:v", "h264_nvenc", "-preset", "p4"] if nvenc_available else ["-c:v", "libx264", "-preset", "fast"]

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-thread_queue_size", "100000",
            "-f", "rawvideo", "-vcodec", "rawvideo", 
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", 
            "-thread_queue_size", "100000",
            "-i", str(input_video_path),
            "-map", "0:v:0", "-map", "1:a:0?", 
            *encoder, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", 
            str(output_video_path)
        ]

        logger.info(f"Starting FFmpeg pipe encoder: {'GPU (NVENC)' if nvenc_available else 'CPU (x264)'}")
        return subprocess.Popen(command, stdin=subprocess.PIPE)