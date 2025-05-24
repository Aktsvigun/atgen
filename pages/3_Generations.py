import json
import os
from pathlib import Path
from time import sleep

import numpy as np
import streamlit as st

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

            # Check if this directory has any iterations with generations
            iter_found = False
            for i in range(10):  # Check first 10 possible iterations
                if (exp_dir / f"iter_{i}" / "generations.json").exists():
                    iter_found = True
                    break

            if iter_found:
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


def main():
    # Set page configuration with icon and title
    st.set_page_config(
        page_title="Model Generations",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add CSS to hide the sidebar only when collapsed
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
        .generation-card {
            background-color: white;
            border-radius: 8px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            border-left: 4px solid #673AB7;
        }
        .generation-text {
            background-color: #f3e5f5;
            border-radius: 4px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            line-height: 1.5;
        }
        .iteration-select {
            background-color: #f0f4f8;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .iteration-badge {
            display: inline-block;
            background-color: #673AB7;
            color: white;
            border-radius: 12px;
            padding: 0.3rem 0.8rem;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
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
        /* Section container to control spacing */
        .section-container {
            margin-bottom: 1rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Page header
    st.markdown(
        "<h1 class='main-header'>💬 Model Generations</h1>", unsafe_allow_html=True
    )

    # Add info box explaining the page
    st.markdown(
        """<div class="info-box">
        <h3>🤖 Model Generated Outputs</h3>
        <p>This page displays sample texts generated by the model during different iterations of the active learning process.
        You can see how the model's outputs evolve as it is trained on more labeled examples.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Add experiment directory selection
    available_workdirs = get_available_workdirs()

    if not available_workdirs:
        st.markdown(
            """<div class="empty-state">
            <h2>No Experiments Found</h2>
            <p>No completed experiments with model generations were found.</p>
            <p>Run an experiment from the configuration page to generate model outputs.</p>
            </div>""",
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

    with st.container():
        NUM_RANDOM_GENS_TO_SHOW = st.slider(
            "Number of examples to display",
            min_value=1,
            max_value=20,
            value=10,
            help="Select how many random generations you want to view",
        )

    # Initialize an empty list to store the data for the DataFrame
    data = []
    column_names = []

    # Initialize iteration number
    iteration_number = 0
    folder_exists = True
    all_iterations = []

    # Iterate through the iterations to find available data
    with st.spinner("Loading model generations..."):
        while folder_exists:
            path = Path(workdir) / f"iter_{iteration_number}" / "generations.json"
            if path.exists():
                sleep(0.5)
                with open(path, "r") as file:
                    json_data = json.load(file)
                    data.append(json_data)
                    all_iterations.append(iteration_number)
                iteration_number += 1
            else:
                folder_exists = False

    # If data is available, display it with iterations
    if data:
        # Create an iteration selector within a container to control spacing
        with st.container():
            st.markdown(
                "<div class='section-container'><h2 class='sub-header'>Select Active Learning Iteration</h2></div>",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="iteration-select">', unsafe_allow_html=True)

            # Display a summary of available iterations
            st.info(
                f"🔍 Found generations from {len(all_iterations)} iterations. Latest iteration: {all_iterations[-1]}"
            )

            # Allow user to select which iteration to view
            selected_iteration = st.selectbox(
                "Choose an iteration to view:",
                options=all_iterations,
                index=len(all_iterations) - 1,  # Default to the latest iteration
                format_func=lambda x: f"Iteration {x}",
            )

            iteration_index = all_iterations.index(selected_iteration)
            st.markdown("</div>", unsafe_allow_html=True)

        # Display the generations from the selected iteration within a container
        with st.container():
            st.markdown(
                f"<div class='section-container'><h2 class='sub-header'>Generated Texts - Iteration {selected_iteration}</h2></div>",
                unsafe_allow_html=True,
            )

            current_data = data[iteration_index]
            data_to_show = np.random.choice(
                current_data, min(len(current_data), NUM_RANDOM_GENS_TO_SHOW), False
            )

            # Create a grid layout for the generations
            cols = st.columns(2)  # Create 2 columns

            for i, text in enumerate(data_to_show):
                col_idx = i % 2  # Alternate between columns
                with cols[col_idx]:
                    cols[col_idx].markdown(
                        f"""<div class="generation-card">
                        <div class="iteration-badge">Generation {i+1}</div>
                        <div class="generation-text">{text}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            """<div class="empty-state">
            <h3>No generations found</h3>
            <p>There are no model generations available at the moment. Please wait for the experiment to produce some outputs.</p>
            </div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
