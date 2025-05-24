from shutil import rmtree
import os
import json
from pathlib import Path
import streamlit as st
from datasets import load_from_disk
from time import sleep
import yaml
import html

from atgen.utils.get_last_workdir import get_last_workdir


def get_available_workdirs():
    """Get a list of all available experiment directories sorted by date (newest first)"""
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

            # Check if this directory has texts to annotate
            annotate_dir = exp_dir / "dataset_to_annotate"
            if annotate_dir.exists():
                all_workdirs.append(exp_dir)

    return all_workdirs


def format_workdir_name(workdir):
    """Format the workdir path into a readable name for the dropdown"""
    # Extract date and experiment name
    parts = str(workdir).split(os.sep)
    if len(parts) >= 3:
        date = parts[-2]
        exp_name = parts[-1]
        return f"{exp_name} ({date})"
    return str(workdir)


def save_temp_annotations(workdir, annotations, current_index):
    """Save annotations to a temporary file"""
    temp_file = workdir / "temp_annotations.json"
    data = {"annotations": annotations, "current_index": current_index}
    with open(temp_file, "w") as f:
        json.dump(data, f)


def load_temp_annotations(workdir):
    """Load annotations from a temporary file if it exists"""
    temp_file = workdir / "temp_annotations.json"
    if temp_file.exists():
        try:
            with open(temp_file, "r") as f:
                data = json.load(f)
            return data.get("annotations", []), data.get("current_index", 0)
        except Exception as e:
            st.warning(f"Failed to load temporary annotations: {e}")
    return [], 0


def delete_temp_annotations(workdir):
    """Delete the temporary annotations file"""
    temp_file = workdir / "temp_annotations.json"
    if temp_file.exists():
        temp_file.unlink()


def main():
    # Set page configuration with icon and title
    st.set_page_config(
        page_title="Human Annotation",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add CSS to hide the sidebar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="false"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Apply custom CSS for better styling
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
        .info-box {
            background-color: #e1f5fe;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #fff8e1;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            border-left: 4px solid #ffa000;
        }
        .success-box {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            border-left: 4px solid #4caf50;
        }
        .empty-state {
            text-align: center;
            padding: 2rem;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin: 2rem 0;
        }
        .text-card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            border-left: 4px solid #ff9800;
        }
        .input-text {
            background-color: #fff3e0;
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            line-height: 1.5;
        }
        .progress-container {
            margin-bottom: 1.5rem;
        }
        .stTextArea>div>div>textarea {
            min-height: 150px !important;
            font-family: 'Courier New', monospace;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            padding: 1rem;
        }
        .stTextArea>div>div>textarea:focus {
            border-color: #1E88E5;
            box-shadow: 0 0 0 1px #1E88E5;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1565C0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        /* Experiment selector styling */
        .experiment-selector {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .system-prompt-box {
            background-color: #e8f5e9;
            padding: 1.5rem;
            border-radius: 5px;
            margin-bottom: 1.5rem;
            border-left: 4px solid #4caf50;
        }
        .template-prompt-box {
            background-color: #e3f2fd;
            padding: 1.5rem;
            border-radius: 5px;
            margin-bottom: 1.5rem;
            border-left: 4px solid #2196f3;
        }
        .system-prompt-content {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 4px;
            line-height: 1.5;
            overflow-x: auto;
            border: 1px solid #e0e0e0;
            font-size: 0.9rem;
        }
        .template-prompt-content {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            line-height: 1.5;
            overflow-x: auto;
            border: 1px solid #e0e0e0;
            font-size: 0.9rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Page header
    st.markdown(
        "<h1 class='main-header'>✍️ Human Annotation</h1>", unsafe_allow_html=True
    )

    # Add info box explaining the page
    st.markdown(
        """
    <div class="info-box">
        <h3>📝 Annotation Interface</h3>
        <p>This page allows human annotators to provide labels for selected examples.
        Your annotations will be used to train and improve the model in subsequent iterations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Add experiment directory selection
    available_workdirs = get_available_workdirs()

    if not available_workdirs:
        st.markdown(
            """
        <div class="empty-state">
            <h2>No Texts Available for Annotation</h2>
            <p>No experiments with texts to annotate were found.</p>
            <p>Run an experiment from the configuration page to generate texts for annotation.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

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

    iter_number = 0

    def reset_session_state():
        st.session_state.annotations = []
        st.session_state.current_index = 0
        delete_temp_annotations(workdir)

    path = workdir / "dataset_to_annotate"
    config_path = workdir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    input_column_name = config["data"]["input_column_name"]

    # Extract system prompt from config if available
    system_prompt = config.get("data", {}).get("system_prompt", None)

    if path.exists():
        st.markdown(
            """
        <div class="warning-box">
            <h3>⚠️ Important Notice</h3>
            <p>Please annotate all texts carefully. Your progress is automatically saved, so you can safely refresh the page if needed.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display system prompt if available
        if system_prompt:
            st.markdown(
                "<h3 class='sub-header'>📋 Prompt Context for Annotation</h3>",
                unsafe_allow_html=True,
            )

            with st.expander("View Prompt Details", expanded=True):
                st.markdown(
                    f"""
                <div class="system-prompt-box">
                    <h3>🤖 System Prompt</h3>
                    <div class="system-prompt-content">{html.escape(system_prompt)}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p><em>Use this prompt context when annotating the texts below.</em></p>",
                    unsafe_allow_html=True,
                )

            st.markdown("<hr>", unsafe_allow_html=True)

        dataset = load_from_disk(path)

        texts = dataset[input_column_name]
        num_texts = len(texts)

        # Initialize or load from temporary file
        if (
            "annotations" not in st.session_state
            or "current_index" not in st.session_state
        ):
            # Try to load from temp file first
            temp_annotations, temp_current_index = load_temp_annotations(workdir)

            # Only use the temp data if it's valid for this dataset
            if temp_current_index <= num_texts and len(temp_annotations) <= num_texts:
                st.session_state.annotations = temp_annotations
                st.session_state.current_index = temp_current_index
                if temp_annotations:
                    st.success(
                        f"Loaded {len(temp_annotations)} saved annotations. You can continue where you left off."
                    )
            else:
                st.session_state.annotations = []
                st.session_state.current_index = 0

        # Display progress information
        if num_texts > 0:
            progress_percentage = (st.session_state.current_index / num_texts) * 100
            st.markdown(
                f"""
            <div class="progress-container">
                <h3 class="sub-header">Annotation Progress</h3>
                <p>Completed {st.session_state.current_index} of {num_texts} texts ({progress_percentage:.1f}%)</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Show progress bar
            st.progress(st.session_state.current_index / num_texts)

        if st.session_state.current_index < num_texts:
            text = texts[st.session_state.current_index]

            st.markdown(
                f"""
            <div class="text-card">
                <h3>Text {st.session_state.current_index + 1}/{num_texts}</h3>
                <div class="input-text">{text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<h3 class='sub-header'>Your Annotation</h3>", unsafe_allow_html=True
            )
            annotation = st.text_area(
                "Please enter your annotation below:",
                key=f"annotation_{st.session_state.current_index}",
                help="Type your annotation for the text above. Be concise and accurate.",
            )

            # Use columns to organize buttons
            col1, col2, col3 = st.columns([1, 1, 1])

            with col2:
                submit_button = st.button(
                    "Save & Continue", type="primary", use_container_width=True
                )

            if submit_button:
                if not annotation.strip():
                    st.error("Please provide an annotation before continuing.")
                else:
                    st.session_state.annotations.append(annotation)
                    st.session_state.current_index += 1
                    # Save to temporary file after each annotation
                    save_temp_annotations(
                        workdir,
                        st.session_state.annotations,
                        st.session_state.current_index,
                    )
                    st.rerun()

        if len(st.session_state.annotations) == num_texts:
            st.markdown(
                """
            <div class="success-box">
                <h3>✅ Annotation Complete!</h3>
                <p>Thank you for completing this batch of annotations.</p>
                <p>Kindly wait for the next batch of texts to be ready for annotation. In the meantime, you can check the model metrics or review its generated texts on the other tabs.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.spinner("Saving your annotations..."):
                dataset = dataset.add_column("annotation", st.session_state.annotations)
                dataset.save_to_disk(workdir / "annotated_query")
                rmtree(path)
                # Clean up temporary file
                delete_temp_annotations(workdir)
                reset_session_state()
                st.success("Your annotations have been saved successfully!")

    else:
        st.markdown(
            """
        <div class="empty-state">
            <h3>No Texts Available for Annotation</h3>
            <p>There are currently no texts available for annotation. If you are an annotator, please wait for the active learning process to select examples for labeling.</p>
            <p>You will be notified when new texts are ready for annotation.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
