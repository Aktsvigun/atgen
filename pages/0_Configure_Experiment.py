import os
import tempfile
import json
import logging
from json.decoder import JSONDecodeError
from datetime import datetime
from omegaconf import OmegaConf

import streamlit as st
import yaml
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from pathlib import Path
from atgen.run_scripts import run_active_learning
from atgen.metrics.deepeval_supported_models_and_metrics import (
    get_available_metrics,
    get_available_models,
)
from atgen.utils.get_last_workdir import get_last_workdir
from atgen.utils.constants import (
    UNLABELED_DATA_SPLIT_DEFAULT_NAME,
    TEST_DATA_SPLIT_DEFAULT_NAME,
)

# Custom CSS for better styling
st.set_page_config(
    page_title="Configure Experiment",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧪",
)

# Constants for experiment status tracking
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_IDLE = "idle"

DATA_SOURCE_UPLOAD = "Upload file"
DATA_SOURCE_HUGGINGFACE = "Huggingface dataset"

# Create a cached resource for experiment status tracking
@st.cache_resource
def get_experiment_status_tracker():
    """Create a shared experiment status tracker across all sessions."""
    return {
        "status": STATUS_IDLE,
        "timestamp": datetime.now().isoformat(),
        "experiment_name": None,
        "experiment_dir": None,
        "current_iteration": 0,
        "total_iterations": 0
    }

# Apply custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-text {
        color: #ff5722;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    div[data-testid="stRadio"] > div {
        padding: 0.3rem;
    }
    div[data-testid="stForm"] {
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #4CAF50;
        color: white;
    }
    .parameter-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .launch-container {
        background-color: #f8f8f8;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .launch-button {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    .launch-button:hover {
        background-color: #388E3C;
        box-shadow: 0 6px 10px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .config-section {
        background-color: transparent;
        margin-top: 0;
        padding-top: 10px;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 1rem;
        border: none;
    }
    .config-header {
        margin-top: 0;
        padding-top: 0;
    }
    /* Style for model selection */
    .model-option {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    .model-option:hover {
        background-color: #f0f7ff;
    }
    .model-option.selected {
        background-color: #e3f2fd;
        border-left: 3px solid #1565C0;
    }
    div[data-testid="stTabsTop"] {
        margin-bottom: -1px !important;
    }
    div[data-testid="stTab"] {
        background-color: #e3f2fd;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: bold;
        margin-right: 5px;
        gap: 10px;
    }
    div[data-testid="stTab"][aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    div[data-testid="stTabContent"] {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding-top: 0;
        margin-top: 0;
    }
    /* Eliminate the white space between tabs and content */
    div[data-testid="stTabContent"] > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    div[data-testid="stTabContent"] > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

LABELLER_MAP = {
    "Open-source / Custom LLM": "custom_llm",
    "API LLM": "api_llm",
    "Human": "human",
    "Golden (only for benchmarking)": "golden",
}


def check_experiment_status():
    """Check if an experiment is currently running."""
    status_tracker = get_experiment_status_tracker()
    if status_tracker.get("status") == STATUS_RUNNING:
        return status_tracker
    return None

def update_experiment_status(status, experiment_name=None, experiment_dir=None, total_iterations=None):
    """Update the experiment status with the current state."""
    status_tracker = get_experiment_status_tracker()
    status_tracker["status"] = status
    status_tracker["timestamp"] = datetime.now().isoformat()
    
    if experiment_name:
        status_tracker["experiment_name"] = experiment_name
    if experiment_dir:
        status_tracker["experiment_dir"] = experiment_dir
    if total_iterations is not None:
        status_tracker["total_iterations"] = total_iterations

    return status_tracker

def create_progress_tracker(progress_container, num_iterations):
    """Create and return a function to update progress tracking."""
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    def update_progress(iteration):
        # Set initial status on first update
        if iteration == 0:
            # Update the status
            update_experiment_status(
                STATUS_RUNNING, 
                total_iterations=num_iterations
            )
            return
            
        if iteration > num_iterations:
            progress = 1.0
        else:
            progress = iteration / num_iterations if num_iterations > 0 else 1.0
        
        progress_bar.progress(progress)
        status_text.text(f"Active Learning Iteration: {iteration}/{num_iterations}")
        
        # Update the status
        update_experiment_status(
            STATUS_RUNNING, 
            total_iterations=num_iterations
        )
    
    return update_progress

def process_uploaded_datasets(train_dataset_path, test_dataset_path, output_dir, config, status):
    """Process uploaded datasets and update config.
    
    Args:
        train_dataset_path: Path to the uploaded training dataset
        test_dataset_path: Path to the uploaded test dataset
        output_dir: Directory to save the processed dataset
        config: Configuration dictionary to update
        status: Streamlit status element for progress updates
        
    Returns:
        Updated configuration dictionary
    """
    # Both files uploaded successfully
    status.info("Processing uploaded datasets...")
    
    # Load train dataset
    if train_dataset_path.endswith(".csv"):
        train_dataset = Dataset.from_csv(train_dataset_path)
    elif train_dataset_path.endswith(".json"):
        train_dataset = Dataset.from_json(train_dataset_path)
    else:
        raise ValueError(
            f"Unsupported file format for train dataset: {train_dataset_path}"
        )

    # Add ID column if not present
    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.add_column(
            "id", list(range(len(train_dataset)))
        )
        
    # Load test dataset
    if test_dataset_path is not None:
        if test_dataset_path.endswith(".csv"):
            test_dataset = Dataset.from_csv(test_dataset_path)
        elif test_dataset_path.endswith(".json"):
            test_dataset = Dataset.from_json(test_dataset_path)
        else:
            raise ValueError(
                f"Unsupported file format for test dataset: {test_dataset_path}"
            )
        if "id" not in test_dataset.column_names:
            test_dataset = test_dataset.add_column(
                "id", list(range(len(test_dataset)))
            )
        dataset_dict = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )
    else:
        st.warning("No test dataset uploaded. Consider uploading the test dataset for better results.")
        test_dataset = None
        dataset_dict = DatasetDict({"train": train_dataset})
    
    dataset_path = os.path.join(output_dir, "dataset_dict")
    dataset_dict.save_to_disk(dataset_path)

    # Set the dataset in the config
    config["data"]["dataset"] = dataset_path
    config["data"]["unlabeled_data_split_name"] = UNLABELED_DATA_SPLIT_DEFAULT_NAME
    config["data"]["test_split_name"] = TEST_DATA_SPLIT_DEFAULT_NAME if test_dataset is not None else None

    status.success("Uploaded datasets processed successfully!")
    
    return config

# Wrapper for run_active_learning to track progress
def run_active_learning_with_progress(config, progress_callback=None):
    """Run active learning with progress tracking"""
    try:
        # Set initial status
        update_experiment_status(
            STATUS_RUNNING,
            experiment_name=config.get("experiment_name", "Active Learning Experiment"),
            experiment_dir=config.get("output_dir", None),
            total_iterations=config["al"]["num_iterations"]
        )
        
        # Call the progress callback immediately to set initial status
        if progress_callback:
            progress_callback(0)
        
        # Run the actual experiment
        result = run_active_learning(config)
        
        # Update status to completed
        update_experiment_status(STATUS_COMPLETED)
        
        return result
    except KeyboardInterrupt:
        update_experiment_status(STATUS_CANCELLED)
        raise
    except Exception as e:
        update_experiment_status(STATUS_FAILED)
        raise

def main():
    # Display header with project logo/title
    st.markdown(
        "<h1 class='main-header'>🧪 Configure New Experiment</h1>",
        unsafe_allow_html=True,
    )

    # Check if an experiment is already running
    running_experiment = check_experiment_status()
    if running_experiment:
        # Recalculate current iteration from directory structure right before displaying status
        if running_experiment.get("experiment_dir"):
            try:
                experiment_dir = running_experiment["experiment_dir"]
                # Check for iteration directories (iter_X) and find the highest X
                iteration_dirs = [d for d in os.listdir(experiment_dir) 
                                if os.path.isdir(os.path.join(experiment_dir, d)) 
                                and d.startswith("iter_")]
                
                if iteration_dirs:
                    # Extract iteration numbers from directory names
                    iteration_numbers = [int(d.split("_")[1]) for d in iteration_dirs if d.split("_")[1].isdigit()]
                    if iteration_numbers:
                        # Current iteration is the highest existing iter_X + 1
                        running_experiment["current_iteration"] = max(iteration_numbers) + 1
                    else:
                        running_experiment["current_iteration"] = 0
                else:
                    running_experiment["current_iteration"] = 0
            except (FileNotFoundError, PermissionError, OSError):
                # In case of any file system errors, keep the current value
                import pdb; pdb.set_trace()

        st.warning(
            f"⚠️ An experiment '{running_experiment.get('experiment_name', 'Unknown')}' is already running! "
            f"Current iteration: {running_experiment.get('current_iteration', '?')}/{running_experiment.get('total_iterations', '?')}. "
            f"The experiment was most likely started by one of the reviewers, so kindly wait for it to finish."
        )
        
        # Show option to force reset the status (in case of stale status)
        if st.button("⚠️ Reset experiment status (Use only if you're sure no experiment is running)"):
            update_experiment_status(STATUS_IDLE)
            st.success("Status reset. Refresh the page to configure a new experiment.")
            st.stop()
        else:
            # Provide navigation to monitoring pages
            st.markdown("You can monitor the experiment progress:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 View Metrics", use_container_width=True):
                    st.switch_page("1_Metrics")
            with col2:
                if st.button("🏷️ View Labeled Examples", use_container_width=True):
                    st.switch_page("2_Labeled_examples")
            with col3:
                if st.button("👩‍🎨 Annotate Examples", use_container_width=True):
                    st.switch_page("3_Annotation")
            st.stop()
            
    st.markdown(
        """
    <div class='info-box'>
    <h3>Experiment Configuration</h3>
    <p>Set up the parameters for your active learning experiment. <br> Complete <strong>all tabs</strong> below,
    then click the "Launch Experiment" button at the bottom to start your run.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create tabs for cleaner organization with more descriptive names
    general_tab, al_tab, labeller_tab, data_tab, model_tab, eval_tab = st.tabs(
        [
            "🔧 General Setup",
            "🎯 Active Learning",
            "👨‍💼 Labelling",
            "📚 Data",
            "🤖 Model & Training",
            "📊 Evaluation",
        ]
    )

    with general_tab:
        st.markdown(
            "<h2 class='config-header'>Experiment Setup</h2>", unsafe_allow_html=True
        )
        st.markdown(
            "<p style='margin-bottom:30px;'>Configure your experiment name and other general settings here.</p>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            experiment_name = st.text_input(
                "🏷️ Experiment Name",
                value="My AL Experiment",
                help="A descriptive name for your experiment",
            )
        with col2:
            seed = st.number_input(
                "🎲 Seed",
                min_value=0,
                max_value=1000000,
                value=42,
                step=1,
                help="Seed for the random number generator",
            )

    with al_tab:
        st.markdown(
            "<h2 class='config-header'>Active Learning Strategy</h2>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            strategy = st.radio(
                "🎯 AL strategy",
                options=["Huds", "Hadas", "IDDS", "Fac-Loc", "Random"],
                help="Choose the active learning strategy to use for data selection",
            ).lower()
            # TODO: testing, remove this
            if strategy == "idds" or strategy == "fac-loc":
                strategy = "hadas"

            query_size = st.number_input(
                "📊 AL query size",
                min_value=1,
                step=1,
                value=10,
                help="Number of examples to select in each active learning iteration",
            )

        with col2:
            with st.expander("🔧 Advanced Strategy Settings", expanded=False):
                strategy_parameters = st.text_area(
                    "Strategy parameters (JSON format):",
                    placeholder='Example: {"embeddings_model_checkpoint: google-bert/bert-large-uncased"}',
                    help="Additional parameters for the active learning strategy in JSON format",
                )

            st.markdown(
                f"""
                <div style="background-color: #f0f7ff; padding: 10px; border-radius: 5px; border-left: 3px solid #1E88E5;">
                    <h4 style="margin-top: 0;">About AL Strategy</h4>
                    <p>The selected strategy will determine how the most informative examples are picked from your dataset for labeling.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<h3 class='section-header'>Stopping Criteria</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("Fill in at least one of the parameters below:")

        col1, col2, col3 = st.columns(3)
        with col1:
            num_iterations = st.number_input(
                "🔄 Number of AL iterations",
                min_value=0,
                max_value=100,
                step=1,
                value=5,
                help="Maximum number of active learning iterations to run",
            )

        with col2:
            required_performance = st.text_area(
                "📈 Required quality (JSON format):",
                placeholder='Example: {"rouge1": 0.5}',
                help="Target quality metrics to achieve before stopping",
            )

        with col3:
            budget = st.number_input(
                "💰 Budget (in $)",
                step=1,
                value=None,
                help="Maximum budget to spend on labeling",
            )

    with labeller_tab:
        st.markdown(
            "<h2 class='config-header'>Labeller Configuration</h2>",
            unsafe_allow_html=True,
        )

        labeller = st.radio(
            "👨‍💼 Labeller",
            [
                "Golden (only for benchmarking)",
                "Open-source / Custom LLM",
                "API LLM",
                "Human",
            ],
            help="Select the type of labeller to use for data annotation",
        )
        labeller = LABELLER_MAP[labeller]

        if labeller == "human":
            price_input_per_example = st.number_input(
                "💵 Price per example (in $)",
                step=0.01,
                value=0.0,
                min_value=0.0,
                max_value=1000.0,
                help="Cost paid to human annotators per example",
            )
        elif labeller == "custom_llm":
            labeller_checkpoint = st.text_input(
                "🤖 Model checkpoint from HuggingFace",
                value="Qwen/Qwen3-32B",
                help="HuggingFace model ID for the custom LLM",
            )
        elif labeller == "api_llm":
            # Add data privacy disclaimer
            st.markdown(
                """
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 15px 0;">
                    <h4 style="color: #856404; margin-top: 0; display: flex; align-items: center;">
                        ⚠️ Data Privacy Notice
                    </h4>
                    <p style="color: #856404; margin-bottom: 0; font-size: 0.9rem;">
                        <strong>Important:</strong> When using API-based labellers (OpenAI, Anthropic, etc.), your dataset will be sent to external services for processing. 
                        Please ensure you have the necessary permissions and that your data complies with the respective service providers' terms of use and privacy policies.
                        Consider using local/custom models if your data contains sensitive or proprietary information.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            col1, col2 = st.columns(2)
            with col1:
                provider = st.radio(
                    "🔌 Provider",
                    ["OpenAI", "Anthropic", "Custom"],
                    help="API provider for the LLM",
                ).lower()
                model = st.text_input(
                    "🤖 Model",
                    value="gpt-4o",
                    help="Model name provided by the API service",
                )

            with col2:
                api_key = st.text_input(
                    "🔑 API key",
                    placeholder="Your API key",
                    type="password",
                    help="API key for the selected provider",
                )
                if provider == "custom":
                    base_url = st.text_input(
                        "🌐 Base URL",
                        placeholder="https://api.openai.com/v1",
                        help="Base URL for custom API endpoint",
                    )

                input_per_1m = st.number_input(
                    "💲 Price per 1M input tokens (in $)",
                    min_value=0.0,
                    step=0.01,
                    help="Cost for 1 million input tokens",
                )
                output_per_1m = st.number_input(
                    "💲 Price per 1M output tokens (in $)",
                    min_value=0.0,
                    step=0.01,
                    help="Cost for 1 million output tokens",
                )

    with data_tab:
        st.markdown(
            "<h2 class='config-header'>Data Configuration</h2>", unsafe_allow_html=True
        )

        data_source = st.radio(
            "📊 Data Source",
            options=[DATA_SOURCE_HUGGINGFACE, DATA_SOURCE_UPLOAD],
            help="Choose where to get your dataset from",
        )

        if data_source == DATA_SOURCE_HUGGINGFACE:
            # Add some explanation text
            st.markdown(
                """
                <div style="background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4 style="margin-top: 0;">💡 Using HuggingFace Datasets</h4>
                    <p>Enter a dataset ID from HuggingFace (e.g., "SpeedOfMagic/gigaword_tiny") or a local path to your dataset.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Dataset input
            dataset = st.text_input(
                "📚 Dataset or path to data",
                value="Yale-LILY/aeslc",
                help="HuggingFace dataset ID or local path to dataset",
            )

            # Split names in two columns
            st.markdown(
                "<p style='margin-top: 15px; margin-bottom: 5px;'><strong>Dataset Split Names</strong></p>",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                unlabeled_data_split_name = st.text_input(
                    "🏋️‍♀️ Unlabeled data split name",
                    value="train",
                    help="Name of the dataset split to use for labelling",
                )
            with col2:
                test_split_name = st.text_input(
                    "🧪 Test split name",
                    value="test",
                    help="Name of the dataset split to use for testing",
                )
        else:
            # Add guidance about file format BEFORE the uploader
            st.markdown(
                """
                <div style="background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4 style="margin-top: 0;">📝 File Format Requirements</h4>
                    <p>Please upload both training and test datasets separately. Each file should contain your dataset with appropriate columns for input and output fields.
                    Make sure to specify the correct column names in the fields below.</p>
                    <ul>
                        <li><strong>CSV files</strong>: Should include header row with column names</li>
                        <li><strong>JSON files</strong>: Should be in a list format where each item is a record with named fields</li>
                    </ul>
                    <p>Example JSON format:</p>
                    <pre style="background-color: #f5f5f5; padding: 5px; font-size: 0.8rem;">[
    {"document": "Text to be summarized", "summary": "Summary text"},
    {"document": "Another text to be summarized", "summary": "Another summary"}
]</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Training Dataset")
                train_uploaded_file = st.file_uploader(
                    "📤 Upload training dataset",
                    type=["csv", "json"],
                    help="Upload a CSV or JSON file to use as your training dataset",
                    key="train_data_uploader",
                )

            with col2:
                st.markdown("#### Test Dataset")
                test_uploaded_file = st.file_uploader(
                    "📤 Upload test dataset",
                    type=["csv", "json"],
                    help="Upload a CSV or JSON file to use as your test dataset",
                    key="test_data_uploader",
                )

            # Handle train dataset
            train_dataset_path = None
            if train_uploaded_file is not None:
                # Create temp file with appropriate extension
                temp_dir = tempfile.mkdtemp()
                file_extension = (
                    ".csv"
                    if train_uploaded_file.name.lower().endswith(".csv")
                    else ".json"
                )
                train_dataset_path = os.path.join(
                    temp_dir, f"train_dataset{file_extension}"
                )

                # Save the uploaded file
                with open(train_dataset_path, "wb") as f:
                    f.write(train_uploaded_file.getvalue())

                st.success(
                    f"Training file '{train_uploaded_file.name}' successfully uploaded!"
                )

            # Handle test dataset
            test_dataset_path = None
            if test_uploaded_file is not None:
                # Create a temporary file to save the uploaded content
                if "temp_dir" not in locals():
                    temp_dir = tempfile.mkdtemp()

                file_extension = (
                    ".csv"
                    if test_uploaded_file.name.lower().endswith(".csv")
                    else ".json"
                )
                test_dataset_path = os.path.join(
                    temp_dir, f"test_dataset{file_extension}"
                )

                # Save the uploaded file
                with open(test_dataset_path, "wb") as f:
                    f.write(test_uploaded_file.getvalue())

                st.success(
                    f"Test file '{test_uploaded_file.name}' successfully uploaded!"
                )
        # Field names section with a visual separator
        st.markdown(
            "<hr style='margin-top: 20px; margin-bottom: 15px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='margin-bottom: 5px;'><strong>Column Names</strong></p>",
            unsafe_allow_html=True,
        )

        # Column definitions for input/output fields
        col1, col2 = st.columns(2)
        with col1:
            input_field = st.text_input(
                "📥 Input field name",
                value="email_body",
                help="Name of the field containing input text in the dataset",
            )
        with col2:
            reference_field = st.text_input(
                "📤 Reference field name",
                value="subject_line",
                help="Name of the field containing reference output in the dataset",
            )

        # Prompt configuration
        st.markdown(
            "<hr style='margin-top: 20px; margin-bottom: 15px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h3 class='section-header'>Prompt Template</h3>", unsafe_allow_html=True
        )
        system_prompt = st.text_area(
            "📝 System prompt",
            placeholder="Summarize the text.",
            value="Summarize the text.",
            help="System prompt for the model. Include few-shot learning examples for better performance.",
        )

    with model_tab:
        st.markdown(
            "<h2 class='config-header'>Model Configuration</h2>", unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            model_checkpoint = st.text_input(
                "🤖 Model checkpoint",
                value="Qwen/Qwen3-1.7B",
                help="HuggingFace model ID for generation",
            )

        with col2:
            st.markdown(
                """
                <div style="background-color: #f8f8f8; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0;">
                    <h4 style="margin-top: 0;">💡 Model Tips</h4>
                    <p>Enter a HuggingFace model ID that's suitable for your task. For best results, use instruction-tuned models when available.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Training parameters section
        st.markdown(
            "<h3 class='section-header'>Training Parameters</h3>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            lora = st.checkbox(
                "🔧 Use LoRA",
                value=True,
                help="Whether to use LoRA for efficient fine-tuning",
            )
            if lora:
                lora_r = st.number_input(
                    "LoRA rank",
                    min_value=1,
                    value=16,
                    step=1,
                    help="LoRA rank parameter",
                )

            learning_rate = st.number_input(
                "📈 Learning rate",
                min_value=0.0,
                value=5e-5,
                format="%e",
                help="Learning rate for training",
            )

            dev_split_size = st.number_input(
                "🔍 Dev split size",
                min_value=0.0,
                value=0.2,
                step=0.01,
                help="Size of the split for validation of the model during training",
            )

        with col2:
            per_device_train_batch_size = st.number_input(
                "⚡ Train batch size",
                min_value=1,
                value=4,
                step=1,
                help="Training batch size per device",
            )

            eval_batch_size = st.number_input(
                "⚡ Evaluation batch size",
                min_value=1,
                value=4,
                step=1,
                help="Batch size for evaluation",
            )

            gradient_accumulation_steps = st.number_input(
                "🔄 Gradient accumulation steps",
                min_value=1,
                value=1,
                step=1,
                help="Number of gradient accumulation steps",
            )

            num_train_epochs = st.number_input(
                "🔄 Number of training epochs",
                min_value=1,
                value=5,
                step=1,
                help="Number of training epochs",
            )

        # Inference parameters section
        st.markdown(
            "<h3 class='section-header'>Inference Parameters</h3>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            max_input_tokens = st.slider(
                "📝 Max input tokens",
                min_value=16,
                max_value=128_000,
                value=4096,
                step=16,
                help="Maximum input sequence length (affects the speed of training)",
            )
            max_new_tokens = st.slider(
                "📏 Max new tokens",
                min_value=1,
                max_value=16_000,
                value=128,
                step=1,
                help="Maximum number of tokens to generate",
            )

        with col2:
            temperature = st.slider(
                "🔥 Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.1,
                help="Temperature for sampling",
            )
            top_p = st.slider(
                "🔝 Top p",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Top p for nucleus sampling",
            )

        with col3:
            framework = st.selectbox(
                "🛠️ Framework",
                options=["vLLM", "SGLang", "Unsloth"],
                index=0,
                help="Framework to use for inference",
            ).lower()
            inference_batch_size = st.number_input(
                "⚡ Inference batch size",
                min_value=1,
                value=4,
                step=1,
                help="Batch size for inference",
            )

    with eval_tab:
        st.markdown(
            "<h2 class='config-header'>Evaluation Configuration</h2>",
            unsafe_allow_html=True,
        )

        metrics = st.multiselect(
            "📊 Metrics to compute",
            [
                "bertscore",
                "bartscore",
                "alignscore",
                "deepeval_answer_relevance",
                "deepeval_faithfulness",
                "deepeval_summarization",
                "deepeval_prompt_alignment",
            ],
            default=["bartscore", "alignscore"],
            help="Metrics to compute for evaluating model performance",
        )

        # Add metrics configuration based on selected metrics
        if metrics:
            # DeepEval configuration
            deepeval_metrics_exist = any(m.startswith("deepeval_") for m in metrics)

            if deepeval_metrics_exist:
                st.markdown(
                    "<h3 class='section-header'>DeepEval Metrics Configuration</h3>",
                    unsafe_allow_html=True,
                )

                # Remove checkbox and always show configuration
                provider = st.selectbox(
                    "Provider",
                    options=["OpenAI", "Anthropic", "OpenRouter", "Custom"],
                    index=0,
                    help="Select the provider for DeepEval metrics",
                )

                if provider == "Custom":
                    evaluation_base_url = st.text_input(
                        "Base URL", help="The base URL for the evaluation provider."
                    )

                evaluation_api_key = st.text_input(
                    "API Key",
                    type="password",
                    help="Required for DeepEval metrics. You can also set the EVALUATION_API_KEY environment variable. If not set, the labeller API key will be used.",
                )

                available_models = [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "claude-3.7-sonnet",
                    "custom",
                ]
                try:
                    available_models = get_available_models(provider) + ["custom"]
                except:
                    pass

                evaluation_model = st.selectbox(
                    "Choose model name",
                    options=available_models,
                    index=0,
                    help="Select the model to use for DeepEval metrics evaluation",
                )

                # Add option for custom model input
                if evaluation_model == "custom":
                    custom_model = st.text_input(
                        "Custom model name",
                        value="",
                        placeholder="e.g., anthropic/claude-3-7-sonnet",
                        help="Enter a custom model identifier",
                    )
                    if custom_model.strip():
                        evaluation_model = custom_model.strip()

                # Additional DeepEval configuration
                with st.expander("Advanced DeepEval Settings", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        deepeval_threshold = st.slider(
                            "Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.01,
                            help="Minimum passing threshold for metrics",
                        )

                        deepeval_include_reason = st.checkbox(
                            "Include Reason",
                            value=True,
                            help="Include explanation for evaluation scores",
                        )

                        deepeval_strict_mode = st.checkbox(
                            "Strict Mode",
                            value=False,
                            help="Enforce binary metric score: 1 for perfection, 0 otherwise",
                        )

                    with col2:
                        deepeval_async_mode = st.checkbox(
                            "Async Mode", value=True, help="Enable concurrent execution"
                        )

                        deepeval_verbose_mode = st.checkbox(
                            "Verbose Mode", value=False, help="Print intermediate steps"
                        )

                        deepeval_truths_extraction_limit = st.number_input(
                            "Truths Extraction Limit",
                            min_value=1,
                            max_value=100,
                            value=10,
                            help="Maximum number of factual truths to extract (for faithfulness metric)",
                        )
            # If no DeepEval metrics are selected, set default values for the configuration
            else:
                deepeval_threshold = 0.5
                deepeval_include_reason = True
                deepeval_strict_mode = False
                deepeval_async_mode = True
                deepeval_verbose_mode = False
                deepeval_truths_extraction_limit = 10

    # Launch button section
    st.markdown(
        """
    <div class="launch-container">
        <h2>Ready to Start Your Experiment?</h2>
        <p>Make sure you've configured all tabs above before proceeding.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Launch button
    if st.button(
        "🚀 Launch Experiment", key="launch_experiment", use_container_width=True
    ):
        # Validate configuration
        with st.spinner("Validating your configuration..."):
            is_valid_required_performance = True
            if required_performance:
                required_performance = required_performance.strip()
                try:
                    required_performance = json.loads(required_performance)
                except JSONDecodeError:
                    is_valid_required_performance = False
                    st.error(
                        "Please provide a valid JSON input for 'required performance' field"
                    )

            # Check if all required fields are filled
            validation_successful = (
                strategy
                and (
                    num_iterations is not None
                    or budget is not None
                    or (required_performance and required_performance != "{}")
                )
                and is_valid_required_performance
            )

        # Only proceed if validation passed
        if validation_successful:
            st.success("Configuration valid! Setting up the experiment...")

            # Check again if an experiment is already running
            if check_experiment_status():
                st.error("Another experiment is already running. Please wait for it to complete.")
                st.stop()
                
            # Create a progress container outside of any spinner
            progress_container = st.container()
            with progress_container:
                st.markdown(
                    """
                <div class='info-box'>
                    <h3>🚀 Experiment is running...</h3>
                    <p>This might take a while. Meanwhile, make yourself some tea with sweets or check the intermediate results, generations of the model, and the labeled examples using the tabs on the left.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                
                # Add progress tracking elements
                progress_bar = st.progress(0)
                progress_status = st.empty()

                # Improved cancel button behavior
                if st.button(
                    "⚠️ Cancel the experiment",
                    key="cancel_experiment",
                    type="secondary",
                ):
                    st.session_state["experiment_cancelled"] = True
                    update_experiment_status(STATUS_CANCELLED)
                    st.warning("Experiment cancelled by user")
                    st.stop()  # Stop execution immediately

            # Initialize cancellation state if not present
            if "experiment_cancelled" not in st.session_state:
                st.session_state["experiment_cancelled"] = False

            # Check for cancellation before proceeding
            if st.session_state["experiment_cancelled"]:
                st.warning("Experiment was cancelled")
                st.session_state["experiment_cancelled"] = False  # Reset the state
                update_experiment_status(STATUS_CANCELLED)
                st.rerun()

            # Create a status element to show progress
            status = st.empty()
            status.info("Initializing experiment...")

            try:
                # Clear Hydra instance if it's already initialized
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                with initialize(config_path="../configs", version_base=None):
                    # Configuration setup code remains unchanged
                    config = compose(
                        config_name="base",
                        overrides=[
                            "data=user_data",
                            f"labeller={labeller}",
                            f"al={strategy}",
                        ],
                    )

                    # Update status message
                    status.info("Configuring dataset parameters...")

                    # Check if we're using uploaded files
                    if data_source == DATA_SOURCE_UPLOAD:
                        # TODO: handle the data here rather than later in the script
                        # Need to use default split names in this case
                        config["data"][
                            "unlabeled_data_split_name"
                        ] = UNLABELED_DATA_SPLIT_DEFAULT_NAME
                        config["data"]["test_split_name"] = TEST_DATA_SPLIT_DEFAULT_NAME
                        # Process uploaded datasets if available
                        if train_dataset_path is not None:
                            config = process_uploaded_datasets(
                                train_dataset_path=train_dataset_path, 
                                test_dataset_path=test_dataset_path, 
                                output_dir=output_dir, 
                                config=config, 
                                status=status
                            )
                        else:
                            st.error("No training dataset uploaded. Please upload a training dataset.")
                            st.stop()
                    else:
                        # Using standard dataset (HuggingFace or local path)
                        config["data"]["dataset"] = dataset
                        config["data"][
                            "unlabeled_data_split_name"
                        ] = unlabeled_data_split_name
                        config["data"]["test_split_name"] = test_split_name

                    config["data"]["input_column_name"] = input_field
                    config["data"]["output_column_name"] = reference_field
                    config["data"]["input_max_length"] = max_input_tokens
                    config["data"]["output_max_length"] = max_new_tokens
                    config["data"]["system_prompt"] = system_prompt

                    # Update status message
                    status.info("Configuring active learning parameters...")

                    # Configure general and AL parameters
                    config["al"]["num_iterations"] = num_iterations
                    if query_size:
                        query_size = int(query_size)
                        config["al"]["query_size"] = query_size
                        config["al"]["init_query_size"] = query_size
                    if budget:
                        config["al"]["budget"] = budget
                    if required_performance:
                        config["al"]["required_performance"] = required_performance

                    # Update status message
                    status.info("Configuring labeller parameters...")

                    # Configure labeller
                    if labeller == "human":
                        if "price_input_per_example" in locals():
                            config["labeller"][
                                "price_input_per_example"
                            ] = price_input_per_example
                    elif labeller == "custom_llm":
                        if "model_checkpoint" in locals():
                            config["labeller"]["model"]["checkpoint"] = labeller_checkpoint
                    elif labeller == "api_llm":
                        config["labeller"]["api_key"] = api_key
                        config["labeller"]["provider"] = provider
                        config["labeller"]["parameters"]["model"] = model
                        if "base_url" in locals() and provider == "custom":
                            config["labeller"]["base_url"] = base_url
                        config["labeller"]["price"]["input_per_1m"] = input_per_1m
                        config["labeller"]["price"]["output_per_1m"] = output_per_1m

                    # Update status message
                    status.info("Configuring model and inference parameters...")

                    # Configure model parameters
                    config["model"]["checkpoint"] = model_checkpoint

                    config["inference"]["framework"] = framework
                    config["inference"]["batch_size"] = inference_batch_size
                    config["inference"]["max_new_tokens"] = max_new_tokens
                    config["inference"]["temperature"] = temperature
                    config["inference"]["top_p"] = top_p

                    # Update status message
                    status.info("Configuring training and evaluation parameters...")

                    # Configure advanced settings
                    if metrics:
                        config["evaluation"]["additional_metrics"] = metrics

                        # Add deepeval configuration to the config if it was configured
                        if (
                            "deepeval_metrics_exist" in locals()
                            and deepeval_metrics_exist
                        ):
                            config["evaluation"]["deepeval"] = {
                                "provider": provider,
                                "api_key": evaluation_api_key,
                                "model": evaluation_model,
                                "threshold": deepeval_threshold,
                                "include_reason": deepeval_include_reason,
                                "strict_mode": deepeval_strict_mode,
                                "async_mode": deepeval_async_mode,
                                "verbose_mode": deepeval_verbose_mode,
                                "truths_extraction_limit": deepeval_truths_extraction_limit,
                            }

                            if (
                                provider == "Custom"
                                and "evaluation_base_url" in locals()
                            ):
                                config["evaluation"]["deepeval"][
                                    "base_url"
                                ] = evaluation_base_url

                        if lora:
                            config["model"]["peft"]["use"] = True
                            config["model"]["peft"]["r"] = lora_r
                        else:
                            config["model"]["peft"]["use"] = False

                        config["training"]["hyperparameters"][
                            "num_epochs"
                        ] = num_train_epochs
                        config["training"]["hyperparameters"][
                            "train_batch_size"
                        ] = per_device_train_batch_size
                        config["training"]["hyperparameters"][
                            "eval_batch_size"
                        ] = eval_batch_size
                        config["training"]["hyperparameters"]["lr"] = learning_rate
                        config["training"]["hyperparameters"][
                            "gradient_accumulation_steps"
                        ] = gradient_accumulation_steps
                        config["training"]["dev_split_size"] = dev_split_size

                    # Update status message
                    status.info("Preparing output directory...")

                    # Set `output_dir` with timestamp
                    current_time = datetime.now()
                    output_dir = os.path.join(
                        "outputs",
                        current_time.strftime("%Y-%m-%d"),
                        f"{experiment_name}_{current_time.strftime('%H-%M-%S')}",
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the configuration
                    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                        f.write(OmegaConf.to_yaml(config))
                    config["output_dir"] = output_dir

                    if labeller == "human":
                        st.markdown(
                            """
                        <div class='info-box' style='background-color: #ffecb3;'>
                            <h3>👋 Human Labelling Selected</h3>
                            <p>For annotator: kindly switch to the tab <strong>Annotation</strong> on the left.</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    # Create progress tracking callback
                    progress_update_callback = create_progress_tracker(
                        progress_container, 
                        num_iterations
                    )

                    # Clear the status before running the experiment
                    status.empty()

                    # Run the active learning experiment with progress tracking
                    with st.spinner("Running active learning experiment..."):
                        run_active_learning_with_progress(
                            config, 
                            progress_callback=progress_update_callback
                        )

                    # Update status to completed
                    update_experiment_status(STATUS_COMPLETED)
                    
                    # Show success indicators after completion
                    st.balloons()

                    # Create success message and navigation options
                    success_container = st.container()
                    with success_container:
                        st.success(
                            "✅ Experiment has finished running! Check the results in the pages below."
                        )

                        # Navigation buttons in columns for better layout
                        nav_col1, nav_col2, nav_col3 = st.columns(3)
                        with nav_col1:
                            if st.button("📊 View Metrics", use_container_width=True):
                                st.switch_page("1_Metrics")
                        with nav_col2:
                            if st.button(
                                "🏷️ View Labeled Examples", use_container_width=True
                            ):
                                st.switch_page("2_Labeled_examples")
                        with nav_col3:
                            if st.button(
                                "👩‍🎨 Annotate Examples", use_container_width=True
                            ):
                                st.switch_page("3_Annotation")
            except Exception as e:
                # Update status to failed
                update_experiment_status(STATUS_FAILED)
                # st.error(f"An error occurred: {str(e)}")
                # import sys, pdb
                # exc_type, exc_value, exc_traceback = sys.exc_info()
                # pdb.post_mortem(exc_traceback)
        elif is_valid_required_performance:
            st.error("❌ You didn't fill one of the required arguments.")


if __name__ == "__main__":
    main()
