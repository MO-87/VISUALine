import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="VISUALine Playground",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "pipeline_configs"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

UPLOAD_EXTENSIONS = [
    "mp4", "mov", "avi", "mkv", "webm",
    "jpg", "jpeg", "png", "webp", "gif",
]

from visualine.core.config_loader import YamlConfigLoader
from visualine.core.logger import setup_logger
from visualine.core.pipeline_manager import PipelineManager


if "logger_setup" not in st.session_state:
    log_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    setup_logger(log_config, LOGS_DIR)
    st.session_state.logger_setup = True

logger = logging.getLogger(__name__)


@st.cache_resource
def get_pipeline_manager():
    config_loader = YamlConfigLoader()
    return PipelineManager(config_loader=config_loader)


def get_available_configs():
    if not CONFIGS_DIR.exists():
        return []

    return sorted([p.name for p in CONFIGS_DIR.glob("*.yaml")])


def get_pipeline_signature(pipeline_config_path: Path):
    """
    Used to know whether the selected YAML changed.

    If the path or modified time changes, we reload the pipeline.
    Otherwise, we reuse the already-loaded long-lived pipeline.
    """
    pipeline_config_path = Path(pipeline_config_path)

    return (
        str(pipeline_config_path.resolve()),
        pipeline_config_path.stat().st_mtime,
    )


def ensure_pipeline_loaded(manager: PipelineManager, pipeline_config_path: Path) -> None:
    """
    Load the pipeline only when needed.
    """
    pipeline_signature = get_pipeline_signature(pipeline_config_path)

    if st.session_state.get("loaded_pipeline_signature") != pipeline_signature:
        manager.load_pipeline(
            pipeline_config_path=pipeline_config_path,
            force_reload=True,
        )
        st.session_state.loaded_pipeline_signature = pipeline_signature
    else:
        logger.info(f"Reusing already-loaded pipeline: {pipeline_config_path.name}")


def resolve_existing_media_path(file_path: Path) -> Path:
    """
    PipelineManager / VideoProcessor may change the actual output suffix.

    Example:
        requested output: /tmp/out.gif
        actual output:    /tmp/out.mp4

    This helper finds the real existing file.
    """
    file_path = Path(file_path)

    if file_path.exists():
        return file_path

    candidate_suffixes = [
        ".mp4", ".mov", ".mkv", ".webm", ".avi",
        ".gif",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    ]

    for suffix in candidate_suffixes:
        candidate = file_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    return file_path


def display_media(file_path: Path, caption: str = "Media") -> None:
    """
    Robust media preview for Streamlit.

    Notes:
    - GIF input is shown with st.image because browser GIF preview is usually more reliable that way.
    - Video output is shown using bytes instead of path for better WSL/temp-file reliability.
    - A download button is added so the result can still be inspected if the browser preview fails.
    """
    file_path = resolve_existing_media_path(Path(file_path))

    st.markdown(f"#### {caption}")

    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return

    suffix = file_path.suffix.lower()

    try:
        media_bytes = file_path.read_bytes()
    except Exception as e:
        st.error(f"Could not read media file: {file_path}")
        st.exception(e)
        return

    if suffix in GIF_EXTENSIONS:
        st.image(media_bytes, caption=caption, width='stretch')
        st.download_button(
            label="⬇️ Download GIF",
            data=media_bytes,
            file_name=file_path.name,
            mime="image/gif",
            width='stretch',
        )
        return

    if suffix in IMAGE_EXTENSIONS:
        st.image(media_bytes, caption=caption, width='stretch')
        st.download_button(
            label="⬇️ Download Image",
            data=media_bytes,
            file_name=file_path.name,
            mime=f"image/{suffix.lstrip('.')}",
            width='stretch',
        )
        return

    if suffix in VIDEO_EXTENSIONS:
        if suffix == ".mp4":
            mime = "video/mp4"
        elif suffix == ".webm":
            mime = "video/webm"
        elif suffix == ".mov":
            mime = "video/quicktime"
        elif suffix == ".avi":
            mime = "video/x-msvideo"
        elif suffix == ".mkv":
            mime = "video/x-matroska"
        else:
            mime = "video/mp4"

        st.video(media_bytes, format=mime)

        st.download_button(
            label="⬇️ Download Video",
            data=media_bytes,
            file_name=file_path.name,
            mime=mime,
            width='stretch',
        )
        return

    st.warning(f"Unsupported preview format: {suffix}")
    st.code(str(file_path))

    st.download_button(
        label="⬇️ Download File",
        data=media_bytes,
        file_name=file_path.name,
        width='stretch',
    )


def write_uploaded_file_to_temp(uploaded_file) -> Path:
    input_suffix = Path(uploaded_file.name).suffix.lower()

    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
    temp_in.write(uploaded_file.read())
    temp_in.close()

    return Path(temp_in.name)


def make_temp_output_path(input_path: Path) -> Path:
    """
    Create a temporary output path.

    Important:
    - GIF is treated as video by PipelineManager.
    - Normal images keep image suffix.
    - Videos use .mp4 as the requested output suffix.
    """
    input_suffix = input_path.suffix.lower()

    is_normal_image = input_suffix in IMAGE_EXTENSIONS
    is_gif = input_suffix in GIF_EXTENSIONS

    if is_normal_image and not is_gif:
        out_suffix = input_suffix
    else:
        out_suffix = ".mp4"

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix)
    temp_out.close()

    return Path(temp_out.name)


st.title("🎬 VISUALine Interactive Playground")
st.markdown("Test your media processing pipelines quickly without editing code.")

with st.sidebar:
    st.header("⚙️ Settings")

    available_configs = get_available_configs()

    if not available_configs:
        st.error(f"No configs found in {CONFIGS_DIR}. Please add some YAML files.")
        st.stop()

    selected_config_name = st.radio(
        "Select Pipeline Configuration:",
        available_configs,
        help="Switching this changes the pipeline instructions.",
    )

    st.divider()

    st.header("📁 Input Media")

    uploaded_file = st.file_uploader(
        "Upload a custom video or image...",
        type=UPLOAD_EXTENSIONS,
    )

    default_input_path = DATA_DIR / "input" / "sample.mp4"

    if uploaded_file is None and not default_input_path.exists():
        st.warning(
            f"No upload provided and default '{default_input_path.name}' was not found."
        )

    st.divider()

    clear_cache_button = st.button(
        "♻️ Clear Pipeline Cache",
        width='stretch',
        help="Use this after changing Python node/model/manager code.",
    )

    if clear_cache_button:
        try:
            manager = get_pipeline_manager()
            manager.teardown(clear_pipeline=True)
        except Exception:
            logger.exception("Failed to teardown manager before clearing cache.")

        st.cache_resource.clear()

        if "loaded_pipeline_signature" in st.session_state:
            del st.session_state.loaded_pipeline_signature

        st.success("Pipeline cache cleared.")

    run_button = st.button(
        "🚀 Run Pipeline",
        width='stretch',
        type="primary",
    )


if run_button:
    if uploaded_file is not None:
        input_path = write_uploaded_file_to_temp(uploaded_file)
    else:
        input_path = default_input_path

    if not input_path.exists():
        st.error(f"Input file not found: {input_path}")
        st.stop()

    output_path = make_temp_output_path(input_path)
    pipeline_config_path = CONFIGS_DIR / selected_config_name

    with st.spinner(f"Running pipeline using `{selected_config_name}`..."):
        try:
            t0 = time.time()

            manager = get_pipeline_manager()

            ensure_pipeline_loaded(
                manager=manager,
                pipeline_config_path=pipeline_config_path,
            )

            manager.run(
                input_path=input_path,
                output_path=output_path,
                teardown_after_run=False,
            )

            # Resolve actual output path in case VideoProcessor changed suffix/container.
            output_path = resolve_existing_media_path(output_path)

            duration = time.time() - t0

            st.success(f"✅ Processing complete in {duration:.2f} seconds!")

            col1, col2 = st.columns(2)

            with col1:
                display_media(input_path, "Original Media")

            with col2:
                display_media(output_path, "Processed Result")

        except Exception as e:
            logger.exception("Pipeline failed.")
            st.error(f"An error occurred: {e}")
            st.exception(e)