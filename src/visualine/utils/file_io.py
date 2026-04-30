import json
import logging
import shutil
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Custom exception for FFmpeg-related I/O errors."""
    pass


class VideoProcessor:
    """
    FFmpeg-based video I/O helper for VISUALine.

    Contract expected by PipelineManager:
    - get_ffmpeg_reader(...) -> process, width, height, fps, total_frames
    - get_ffmpeg_writer(...) -> process, actual_output_path
    - mux_audio(...)
    """

    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.ffprobe_path = shutil.which("ffprobe")

        if not self.ffmpeg_path or not self.ffprobe_path:
            raise FFmpegError("FFmpeg and ffprobe must be installed and available in PATH.")

        self._available_encoders = self._get_available_encoders()

    def _get_available_encoders(self) -> str:
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except Exception as e:
            logger.warning(f"Failed to query FFmpeg encoders: {e}")
            return ""

    @staticmethod
    def _parse_fraction(value: Optional[str], default: float = 30.0) -> float:
        if not value or value in {"N/A", "0/0"}:
            return default

        try:
            parsed = float(Fraction(value))
            return parsed if parsed > 0 else default
        except Exception:
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        if value is None or value == "N/A":
            return default

        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        if value is None or value == "N/A":
            return default

        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _start_stderr_logger(pipe, prefix: str) -> None:
        if pipe is None:
            return

        def _log_pipe():
            try:
                for line in iter(pipe.readline, b""):
                    msg = line.decode("utf-8", errors="replace").strip()
                    if msg:
                        logger.warning(f"{prefix}: {msg}")
            except Exception:
                pass

        threading.Thread(target=_log_pipe, daemon=True).start()

    def _get_encoder_flags(self, width: int, height: int, has_alpha: bool = False) -> list:
        """
        Pick a practical encoder.

        Notes:
        - H.264 is used for normal resolutions.
        - HEVC is used when dimensions exceed common H.264 limits.
        - ProRes 4444 is used for alpha output.
        """

        if has_alpha:
            logger.info("Alpha channel detected. Using ProRes 4444 in MOV container.")
            return [
                "-c:v", "prores_ks",
                "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le",
            ]

        needs_hevc = width > 4096 or height > 2304
        encoders = self._available_encoders

        if needs_hevc:
            logger.info(f"Resolution {width}x{height} is large. Using HEVC when available.")

            if "hevc_nvenc" in encoders:
                return [
                    "-c:v", "hevc_nvenc",
                    "-preset", "p4",
                    "-tune", "hq",
                    "-cq", "26",
                    "-pix_fmt", "yuv420p",
                ]

            if "hevc_videotoolbox" in encoders and sys.platform == "darwin":
                return [
                    "-c:v", "hevc_videotoolbox",
                    "-q:v", "55",
                    "-pix_fmt", "yuv420p",
                ]

            if "hevc_amf" in encoders:
                return [
                    "-c:v", "hevc_amf",
                    "-rc", "cqp",
                    "-qp_p", "26",
                    "-qp_i", "26",
                    "-quality", "quality",
                    "-pix_fmt", "yuv420p",
                ]

            if "hevc_qsv" in encoders:
                return [
                    "-c:v", "hevc_qsv",
                    "-global_quality", "26",
                    "-pix_fmt", "yuv420p",
                ]

            return [
                "-c:v", "libx265",
                "-preset", "fast",
                "-crf", "26",
                "-pix_fmt", "yuv420p",
            ]

        if "h264_nvenc" in encoders:
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-tune", "hq",
                "-cq", "26",
                "-pix_fmt", "yuv420p",
            ]

        if "h264_videotoolbox" in encoders and sys.platform == "darwin":
            return [
                "-c:v", "h264_videotoolbox",
                "-q:v", "55",
                "-pix_fmt", "yuv420p",
            ]

        if "h264_amf" in encoders:
            return [
                "-c:v", "h264_amf",
                "-rc", "cqp",
                "-qp_p", "26",
                "-qp_i", "26",
                "-quality", "quality",
                "-pix_fmt", "yuv420p",
            ]

        if "h264_qsv" in encoders:
            return [
                "-c:v", "h264_qsv",
                "-global_quality", "26",
                "-pix_fmt", "yuv420p",
            ]

        return [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
        ]

    def get_video_info(self, video_path: Path) -> dict:
        """
        Extract basic video metadata.

        Returns:
            {
                "width": int,
                "height": int,
                "fps": float,
                "frames": int
            }
        """

        video_path = Path(video_path)

        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of", "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            streams = data.get("streams", [])
            if not streams:
                raise FFmpegError(f"No video stream found in: {video_path}")

            stream = streams[0]
            fmt = data.get("format", {})

            width = self._safe_int(stream.get("width"))
            height = self._safe_int(stream.get("height"))

            if width <= 0 or height <= 0:
                raise FFmpegError(f"Invalid video dimensions for {video_path}: {width}x{height}")

            fps = self._parse_fraction(
                stream.get("avg_frame_rate") or stream.get("r_frame_rate"),
                default=30.0,
            )

            frames = self._safe_int(stream.get("nb_frames"), default=0)

            if frames <= 0:
                duration = self._safe_float(stream.get("duration"), default=0.0)
                if duration <= 0:
                    duration = self._safe_float(fmt.get("duration"), default=0.0)

                if duration > 0 and fps > 0:
                    frames = int(round(duration * fps))

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
            }

        except Exception as e:
            logger.error(f"Failed to extract video info for {video_path}: {e}")
            raise

    def get_ffmpeg_reader(
        self,
        input_video_path: Path,
        max_dimension: int = 1920,
        use_hwaccel: bool = False,
    ):
        """
        Create an FFmpeg raw RGB reader pipe.

        Returns:
            process, width, height, fps, total_frames
        """

        input_video_path = Path(input_video_path)
        info = self.get_video_info(input_video_path)

        orig_w = info["width"]
        orig_h = info["height"]

        if max(orig_w, orig_h) > max_dimension:
            scale = max_dimension / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Keep dimensions even for encoder compatibility.
            new_w = max(2, new_w - (new_w % 2))
            new_h = max(2, new_h - (new_h % 2))

            logger.info(
                f"Downscaling input from {orig_w}x{orig_h} "
                f"to {new_w}x{new_h} for AI processing."
            )
        else:
            new_w, new_h = orig_w, orig_h

        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
        ]

        # Disabled by default because auto hwaccel can be unstable on some systems.
        if use_hwaccel:
            command += ["-hwaccel", "auto"]

        command += [
            "-i", str(input_video_path),
            "-vf", f"scale={new_w}:{new_h}:flags=lanczos",
            "-an",
            "-sn",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-",
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._start_stderr_logger(process.stderr, "FFmpeg Reader")

        return process, new_w, new_h, info["fps"], info.get("frames", 0)

    def get_ffmpeg_writer(
        self,
        output_video_path: Path,
        width: int,
        height: int,
        fps: float,
        channels: int = 3,
    ):
        """
        Create an FFmpeg raw video writer pipe.

        Returns:
            process, actual_output_path
        """

        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        has_alpha = channels == 4

        if has_alpha and output_video_path.suffix.lower() != ".mov":
            logger.warning(
                f"Alpha output detected. Changing container from "
                f"{output_video_path.suffix} to .mov."
            )
            output_video_path = output_video_path.with_suffix(".mov")

        if not has_alpha:
            compatible_containers = {".mp4", ".mov", ".mkv", ".avi"}

            if output_video_path.suffix.lower() not in compatible_containers:
                logger.warning(
                    f"Output container {output_video_path.suffix} may not support "
                    f"the selected encoder. Changing output to .mp4."
                )
                output_video_path = output_video_path.with_suffix(".mp4")

        input_pix_fmt = "rgba" if has_alpha else "rgb24"
        encoder_flags = self._get_encoder_flags(width, height, has_alpha)

        command = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel", "warning",

            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", input_pix_fmt,
            "-r", f"{fps:.6f}",
            "-i", "-",

            *encoder_flags,

            # Color metadata for normal video players.
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
        ]

        if output_video_path.suffix.lower() in {".mp4", ".mov"}:
            command += ["-movflags", "+faststart"]

        command += [str(output_video_path)]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._start_stderr_logger(process.stderr, "FFmpeg Writer")

        return process, output_video_path

    def mux_audio(
        self,
        original_video_path: Path,
        silent_video_path: Path,
        final_output_path: Path,
    ) -> None:
        """
        Copy audio from the original video into the processed silent video.
        """

        original_video_path = Path(original_video_path)
        silent_video_path = Path(silent_video_path)
        final_output_path = Path(final_output_path)

        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Muxing original audio into processed video...")

        command = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel", "error",

            "-i", str(silent_video_path),
            "-i", str(original_video_path),

            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c", "copy",
            "-shortest",

            str(final_output_path),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise FFmpegError(
                "Failed to mux audio.\n"
                f"Command stderr:\n{result.stderr.strip()}"
            )