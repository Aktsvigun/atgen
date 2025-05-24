"""
Labeled Examples Viewer

This page displays examples that have been selected and labeled during the active learning process.
It shows system prompts, few-shot examples (if available), and a random sample of labeled examples.
"""

import os
import json
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from datasets import load_from_disk
import traceback as import_traceback

from atgen.utils.get_last_workdir import get_last_workdir
from atgen.utils.constants import MESSAGES_COLUMN_NAME


# --------------------------
# EXPERIMENT DIRECTORY UTILS
# --------------------------


def get_available_workdirs():
    """
    Find all experiment directories that contain labeled examples.

    Returns:
        list: List of Path objects to experiment directories, sorted by date (newest first)
    """
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []

    all_workdirs = []

    # Loop through each date directory
    for date_dir in sorted(outputs_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue

        # Loop through experiment directories within the date
        for exp_dir in sorted(date_dir.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue

            # Check if this directory has query data
            query_dir = exp_dir / "query"
            if query_dir.exists():
                all_workdirs.append(exp_dir)

    return all_workdirs


def format_workdir_name(workdir):
    """
    Format the workspace directory path into a readable name for the dropdown.

    Args:
        workdir (Path): Path to the workspace directory

    Returns:
        str: Formatted name like "experiment_name (date)"
    """
    parts = str(workdir).split(os.sep)
    if len(parts) >= 3:
        date = parts[-2]
        exp_name = parts[-1]
        return f"{exp_name} ({date})"
    return str(workdir)


# --------------------------
# FEW-SHOT EXAMPLES HANDLING
# --------------------------


def get_few_shot_examples_from_workdir(workdir, dataset, config):
    """
    Load few-shot examples from the workspace directory.

    Args:
        workdir (Path): Path to the workspace directory
        dataset (Dataset): The dataset containing examples
        config (OmegaConf): Configuration dictionary

    Returns:
        list: List of few-shot example messages or None if not found
    """
    few_shot_examples_dir = workdir / "few_shot_examples"

    if not few_shot_examples_dir.exists():
        return None

    try:
        # Load the few-shot examples dataset directly
        few_shot_dataset = load_from_disk(str(few_shot_examples_dir))

        if len(few_shot_dataset) == 0:
            return None

        # Get examples from the dataset
        few_shot_examples = []
        input_column_name = config.get("data", {}).get("input_column_name", "")
        output_column_name = config.get("data", {}).get("output_column_name", "")

        for i in range(len(few_shot_dataset)):
            example = few_shot_dataset[i]
            # Get input and output for this example
            input_text = example.get(input_column_name, "")
            output_text = example.get(output_column_name, "")

            # Create user and assistant messages based on input/output
            user_message = {"role": "user", "content": input_text}
            assistant_message = {"role": "assistant", "content": output_text}

            # Add to few-shot examples if both are found
            if input_text and output_text:
                few_shot_examples.append(user_message)
                few_shot_examples.append(assistant_message)

        return few_shot_examples if few_shot_examples else None
    except Exception as e:
        st.error(f"Error loading few-shot examples: {str(e)}")
        return None


# --------------------------
# UI COMPONENTS
# --------------------------


def setup_page():
    """
    Set up the page configuration and apply custom CSS.
    """
    # Set page configuration with icon and title
    st.set_page_config(
        page_title="Labeled Examples",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Page header
    st.markdown(
        "<h1 class='main-header'>📋 Labeled Examples</h1>", unsafe_allow_html=True
    )

    # Add info box explaining the labeled examples page - remove unnecessary line breaks
    st.markdown(
        """<div class="info-box">
        <h3>📄 Active Learning Labeled Examples</h3>
        <p>This page displays examples that have been selected and labeled during the active learning process.
        These examples provide insight into what kinds of data are being used to train and improve the model.</p>
        </div>""",
        unsafe_allow_html=True,
    )


def display_workdir_selector(available_workdirs):
    """
    Display a dropdown to select the experiment directory.

    Args:
        available_workdirs (list): List of available experiment directories

    Returns:
        Path: Selected workspace directory
    """
    # Create a dictionary for the dropdown options
    workdir_options = {format_workdir_name(wd): wd for wd in available_workdirs}

    # Try to get the last workdir
    try:
        last_workdir = get_last_workdir()
        default_option = format_workdir_name(last_workdir)
    except:
        default_option = format_workdir_name(available_workdirs[0])

    # Only show experiment selector if there's more than one experiment
    if len(available_workdirs) > 1:
        st.markdown('<div class="experiment-selector">', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_option = st.selectbox(
                "📂 Select experiment directory:",
                options=list(workdir_options.keys()),
                index=(
                    list(workdir_options.keys()).index(default_option)
                    if default_option in workdir_options
                    else 0
                ),
            )
            workdir = workdir_options[selected_option]

        with col2:
            # Add a refresh button
            if st.button(
                "🔄 Refresh list", help="Refresh the list of available experiments"
            ):
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        workdir = available_workdirs[0]
        st.markdown(
            f"<div style='font-size:0.9rem; color:#666; margin-bottom:1rem;'>📂 Experiment directory: <code>{format_workdir_name(workdir)}</code></div>",
            unsafe_allow_html=True,
        )

    return workdir


def display_system_prompt(system_prompt):
    """
    Display the system prompt in a styled container.

    Args:
        system_prompt (str): The system prompt text
    """
    with st.container():
        st.markdown(
            f"""<div class="section-container">
            <h2 class='sub-header'>System Prompt</h2>
            <div class="system-prompt-container">
                <div class="system-prompt-header">🧠 System Instructions</div>
                <div class="system-prompt-text">{system_prompt}</div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )


def display_few_shot_examples(workdir, dataset, config):
    """
    Display few-shot examples in a styled container.

    Args:
        workdir (Path): Path to the workspace directory
        dataset (Dataset): The dataset containing examples
        config (OmegaConf): Configuration dictionary
    """
    few_shot_examples = get_few_shot_examples_from_workdir(workdir, dataset, config)
    if not few_shot_examples:
        return

    # Check if we have any valid examples
    if len(few_shot_examples) == 0:
        return

    # Container for few-shot examples section to control spacing
    with st.container():
        st.markdown(
            """<div class="section-container">
            <h2 class='sub-header'>Few-Shot Examples</h2>
            <div class="few-shot-container">""",
            unsafe_allow_html=True,
        )

        for i, msg in enumerate(few_shot_examples):
            role = msg.get("role", "").lower()
            # Process content to ensure proper display
            content = msg.get("content", "")
            if content:
                # Ensure content is properly stripped and formatted
                content = content.strip()

            # Skip if content is empty
            if not content:
                continue

            if role == "user":
                st.markdown(
                    f"""<div class="few-shot-message user-message">
                    <div class="user-prompt-header">🧒 User</div>
                    <div class="user-prompt-text">{content}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif role == "assistant":
                st.markdown(
                    f"""<div class="few-shot-message assistant-message">
                    <div class="assistant-prompt-header">🧙‍♂️ Assistant</div>
                    <div class="assistant-prompt-text">{content}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif role == "system":
                st.markdown(
                    f"""<div class="few-shot-message system-message">
                    <div class="system-prompt-header">🧠 System</div>
                    <div class="system-prompt-text">{content}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("</div></div>", unsafe_allow_html=True)


def display_examples(dataset, input_column_name, output_column_name, num_examples):
    """
    Display a random sample of examples from the dataset.

    Args:
        dataset (Dataset): The dataset containing examples
        input_column_name (str): Column name for inputs
        output_column_name (str): Column name for outputs
        num_examples (int): Number of examples to display
    """
    with st.container():
        st.markdown(
            """<div class="section-container">
            <h2 class='sub-header'>Random Sample of Labeled Examples</h2>
            </div>""",
            unsafe_allow_html=True,
        )

        # Get random sample of examples
        random_idx = np.random.choice(
            range(len(dataset)),
            min(len(dataset), num_examples),
            False,
        )
        subset = dataset[random_idx]
        inputs = subset[input_column_name]
        outputs = subset[output_column_name]

        # Display each example
        for i in range(len(inputs)):
            st.markdown(
                f"""<div class="example-card">
                <div class="card-header">Example #{i+1}</div>
                <div class="card-header">Input:</div>
                <div class="input-text">{inputs[i]}</div>
                <div class="card-header">Annotation:</div>
                <div class="annotation-text">{outputs[i]}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def load_dataset(workdir):
    """
    Load the dataset from the workspace directory.

    Args:
        workdir (Path): Path to the workspace directory

    Returns:
        tuple: (dataset, config) or (None, None) if no dataset found
    """
    # Load config
    try:
        with open(workdir / "config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return None, None

    # Load dataset
    dataset_path = workdir / "query"

    if not dataset_path.exists():
        st.markdown(
            """
            <div class="empty-state">
                <h3>No labeled examples available</h3>
                <p>There are no labeled examples available at the moment. Please wait for the experiment to select and label some examples.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None, config

    try:
        with st.spinner("Loading labeled examples..."):
            dataset = load_from_disk(str(dataset_path))

            if len(dataset) == 0:
                st.markdown(
                    """
                    <div class="empty-state">
                        <h3>No examples found</h3>
                        <p>The dataset exists but contains no examples.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return None, config

            return dataset, config
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, config


# --------------------------
# MAIN APPLICATION LOGIC
# --------------------------


def main():
    """Main application entry point."""

    # Setup the page with styling and headers
    setup_page()

    # Get available experiment directories
    available_workdirs = get_available_workdirs()

    if not available_workdirs:
        st.markdown(
            """<div class="empty-state">
            <h2>No Experiments Found</h2>
            <p>No completed experiments with labeled examples were found.</p>
            <p>Run an experiment from the configuration page to generate labeled examples.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Display experiment selector and get selected workdir
    workdir = display_workdir_selector(available_workdirs)

    # Load dataset and config
    dataset, config = load_dataset(workdir)
    if not dataset:
        return

    # Extract configuration
    input_column_name = config.get("data", {}).get("input_column_name")
    output_column_name = config.get("data", {}).get("output_column_name")
    system_prompt = config.get("data", {}).get(
        "system_prompt", "No system prompt found in config"
    )

    # Show dataset statistics in a single container to avoid spacing issues
    with st.container():
        st.markdown(
            f"""<div class="section-container">
            <h2 class='sub-header'>Dataset Overview</h2>
            </div>""",
            unsafe_allow_html=True,
        )
        st.info(f"Total number of labeled examples: {len(dataset)}")

    # Display number of examples selector
    num_examples = st.slider(
        "Number of examples to display",
        min_value=1,
        max_value=10,
        value=5,
        help="Select how many random examples you want to view",
    )

    # Display system prompt
    display_system_prompt(system_prompt)

    # Display few-shot examples if available
    display_few_shot_examples(workdir, dataset, config)

    # Display random sample of examples
    display_examples(dataset, input_column_name, output_column_name, num_examples)


# --------------------------
# CSS STYLING
# --------------------------


def get_custom_css():
    """
    Return custom CSS for styling the page.
    """
    return """
    <style>
    /* Only hide sidebar when collapsed */
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: none;
    }
    /* Header styles */
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
    /* Card and container styles */
    .info-box {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    /* Fix for empty space in info box */
    .info-box p {
        margin-bottom: 0;
    }
    .empty-state {
        text-align: center;
        padding: 2rem;
        background-color: #f5f5f5;
        border-radius: 5px;
        margin: 2rem 0;
    }
    .example-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1.5rem;
    }
    .card-header {
        font-weight: 600;
        color: #1976D2;
        margin-bottom: 0.5rem;
    }
    .input-text {
        background-color: #e8f0fe;
        padding: 0.75rem;
        border-radius: 4px;
        white-space: pre-wrap;
        margin-bottom: 1rem;
    }
    .annotation-text {
        background-color: #e0f2f1;
        padding: 0.75rem;
        border-radius: 4px;
        white-space: pre-wrap;
        margin-bottom: 0.5rem;
    }
    /* System prompt styling */
    .system-prompt-container {
        background-color: #f0f4c3;
        border-radius: 5px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-left: 4px solid #afb42b;
    }
    .system-prompt-header {
        font-weight: 600;
        color: #827717;
        margin-bottom: 0.5rem;
    }
    .system-prompt-text {
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 0.9rem;
    }
    /* Few-shot examples styling */
    .few-shot-container {
        background-color: #e8eaf6;
        border-radius: 5px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3949ab;
    }
    /* Condensed section margins */
    .section-container {
        margin-bottom: 1rem;
    }
    .few-shot-header {
        font-weight: 600;
        color: #283593;
        margin-bottom: 0.5rem;
    }
    .few-shot-message {
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-radius: 4px;
        background-color: #f5f5f5;
    }
    .few-shot-message:last-child {
        margin-bottom: 0;  /* Remove margin from last message */
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 3px solid #8bc34a;
    }
    .system-message {
        background-color: #fff8e1;
        border-left: 3px solid #ffc107;
    }
    /* Message header styling */
    .user-prompt-header, .assistant-prompt-header, .system-prompt-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    /* Message text styling */
    .user-prompt-text, .assistant-prompt-text {
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 0.9rem;
    }
    /* Fix for stMarkdown empty paragraphs */
    p:empty {
        display: none !important;
    }
    /* Remove extra padding in st elements */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    div.stMarkdown > div > p {
        margin-bottom: 0 !important;
    }
    .element-container {
        margin-bottom: 0 !important;
    }
    /* Fix for empty divs */
    div:empty {
        display: none !important;
    }
    /* Remove default streamlit spacing */
    .stMarkdown div[data-testid="stVerticalBlock"] > div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Tab styling */
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
    /* Remove whitespace between tabs and content */
    div[data-testid="stTabContent"] > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    div[data-testid="stTabContent"] > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Experiment selector styling */
    .experiment-selector {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """


if __name__ == "__main__":
    main()
