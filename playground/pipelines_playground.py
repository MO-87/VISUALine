import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="VISUALine Playground",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "pipeline_configs"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

from visualine.core.config_loader import YamlConfigLoader
from visualine.core.logger import setup_logger
from visualine.core.pipeline_manager import PipelineManager

if "logger_setup" not in st.session_state:
    log_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    return [p.name for p in CONFIGS_DIR.glob("*.yaml")]

st.title("🎬 VISUALine Interactive Playground")
st.markdown("Test your video processing pipelines quickly without editing code.")

with st.sidebar:
    st.header("⚙️ Settings")
    
    available_configs = get_available_configs()
    if not available_configs:
        st.error(f"No configs found in {CONFIGS_DIR}. Please add some YAML files.")
        st.stop()
        
    selected_config_name = st.radio(
        "Select Pipeline Configuration:",
        available_configs,
        help="Switching this instantly changes the pipeline instructions."
    )
    
    st.divider()
    
    st.header("📁 Input Video")
    uploaded_file = st.file_uploader("Upload a custom video...", type=["mp4", "mov", "avi"])
    
    default_input_path = DATA_DIR / "input" / "sample.mp4"
    if not uploaded_file and not default_input_path.exists():
        st.warning(f"No upload provided and default '{default_input_path.name}' not found.")
        
    st.divider()
    run_button = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

if run_button:
    if uploaded_file is not None:
        input_suffix = Path(uploaded_file.name).suffix
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
        temp_in.write(uploaded_file.read())
        temp_in.close()
        input_video_path = Path(temp_in.name)
    else:
        input_video_path = default_input_path

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_out.close()
    output_video_path = Path(temp_out.name)
    
    pipeline_config_path = CONFIGS_DIR / selected_config_name
    
    with st.spinner(f"Running pipeline using `{selected_config_name}`..."):
        try:
            t0 = time.time()
            manager = get_pipeline_manager()
            manager.load_pipeline(pipeline_config_path=pipeline_config_path)
            
            manager.run(input_path=input_video_path, output_path=output_video_path)
            
            duration = time.time() - t0
            st.success(f"✅ Processing complete in {duration:.2f} seconds!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                with open(input_video_path, 'rb') as f:
                    st.video(f.read())
                    
            with col2:
                st.subheader("Processed Result")
                with open(output_video_path, 'rb') as f:
                    st.video(f.read())
                
        except Exception as e:
            logger.exception("Pipeline failed.")
            st.error(f"An error occurred: {e}")
            st.exception(e)