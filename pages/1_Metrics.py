from time import sleep
import io
import base64
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from atgen.visualize.plot_line import plot_line
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

            # Check if this directory has metrics
            metrics_file = exp_dir / "metrics.csv"
            if metrics_file.exists():
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


def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generates a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def get_figure_download_link(fig, filename="figure.png", text="Download PNG"):
    """Generates a link to download a matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href


def generate_pdf_report(df, chart, metric_name, workdir):
    """Generate a simple PDF report with metrics data"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        Image,
    )
    from reportlab.lib.styles import getSampleStyleSheet
    import numpy as np

    # Create a temporary file for the image
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        chart_path = temp_file.name

    # Save the plotly chart as an image
    chart.write_image(chart_path, width=800, height=400)

    # Create a PDF buffer
    buffer = io.BytesIO()

    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Add a title
    styles = getSampleStyleSheet()
    elements.append(
        Paragraph(f"Active Learning Metrics Report - {metric_name}", styles["Title"])
    )
    elements.append(Spacer(1, 20))

    # Add experiment info
    elements.append(Paragraph(f"Experiment path: {workdir}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # Add the chart
    elements.append(Image(chart_path, width=450, height=250))
    elements.append(Spacer(1, 20))

    # Add a table with the metrics data
    data = [["Iteration"] + [col for col in df.columns if col in [metric_name]]]
    for i, row in df.iterrows():
        data.append([i] + [row[col] for col in df.columns if col in [metric_name]])

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    elements.append(table)

    # Add a summary
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Summary", styles["Heading2"]))

    if len(df) > 1:
        initial_value = df[metric_name].iloc[0]
        final_value = df[metric_name].iloc[-1]
        improvement = final_value - initial_value
        percent_improvement = (
            (improvement / initial_value * 100) if initial_value != 0 else float("inf")
        )

        elements.append(
            Paragraph(f"Initial value: {initial_value:.4f}", styles["Normal"])
        )
        elements.append(Paragraph(f"Final value: {final_value:.4f}", styles["Normal"]))
        elements.append(
            Paragraph(f"Absolute improvement: {improvement:.4f}", styles["Normal"])
        )
        if not np.isinf(percent_improvement):
            elements.append(
                Paragraph(
                    f"Relative improvement: {percent_improvement:.2f}%",
                    styles["Normal"],
                )
            )

    # Build the PDF
    doc.build(elements)

    # Clean up the temporary file
    os.unlink(chart_path)

    # Return the buffer value
    buffer.seek(0)
    return buffer


def main():
    # Set page configuration with icon and title
    st.set_page_config(
        page_title="Metrics",
        page_icon="📊",
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
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #e1f5fe;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .empty-state {
            text-align: center;
            padding: 2rem;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin: 2rem 0;
        }
        .metric-selector {
            background-color: #f0f4f8;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        /* Radio button styles */
        div[data-testid="stRadio"] > div {
            padding: 0.3rem;
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
            background-color: transparent;
            border: none;
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
        /* Export options styling */
        .export-section {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
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
    """,
        unsafe_allow_html=True,
    )

    # Page header
    st.markdown(
        "<h1 class='main-header'>📊 Performance Metrics</h1>", unsafe_allow_html=True
    )

    METRICS_MAP = {
        "BARTScore s->h": "BARTScore-sh",
        "BARTScore h->r": "BARTScore-hr",
        "AlignScore": "alignscore",
        "ROUGE-1": "rouge1",
        "ROUGE-2": "rouge2",
        "ROUGE-L": "rougeL",
        # DeepEval metrics
        "Answer Relevance": "deepeval_answer_relevance",
        "Faithfulness": "deepeval_faithfulness",
        "Summarization": "deepeval_summarization",
        "Prompt Alignment": "deepeval_prompt_alignment",
    }

    # Add experiment directory selection
    available_workdirs = get_available_workdirs()

    if not available_workdirs:
        st.markdown(
            """
        <div class="empty-state">
            <h2>No Experiments Found</h2>
            <p>No completed experiments with metrics were found.</p>
            <p>Run an experiment from the configuration page to generate metrics.</p>
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

    # Add info box explaining the metrics page
    st.markdown(
        """
    <div class="info-box">
        <h3>📈 Active Learning Evaluation Metrics</h3>
        <p>This page displays the performance metrics for each iteration of the active learning process.
        You can track how the model's performance improves as more examples are added to the training set.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize an empty DataFrame
    df = None

    path = Path(workdir) / "metrics.csv"

    if path.exists():
        with st.spinner("Loading metrics data..."):
            # Give it a few seconds to sleep: maybe it is being saved now
            sleep(1)
            df = pd.read_csv(path)
            # Ensure we explicitly filter out all reason fields
            df = df.loc[
                :,
                [
                    col
                    for col in df.columns
                    if not col.startswith("time_")
                    and not col.endswith("_reasons")
                    and not "reason" in col.lower()
                ],
            ]

    # If data is available, create a DataFrame
    if df is not None:
        # Set 'iteration' as the index
        df.index.name = "Iteration"
        st.markdown("<h2 class='sub-header'>Metrics Table</h2>", unsafe_allow_html=True)

        # Format the dataframe with styling
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(
            df.style.highlight_max(axis=0, color="#e6f7ff").format(precision=4),
            use_container_width=True,
            height=300,
        )

        # Add export buttons for the table
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                get_table_download_link(df, "metrics.csv", "📥 Download CSV"),
                unsafe_allow_html=True,
            )
        with col2:
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=True)
            excel_buffer.seek(0)
            excel_b64 = base64.b64encode(excel_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}" download="metrics.xlsx">📥 Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
        with col3:
            # Add JSON export
            json_buffer = io.StringIO()
            df.to_json(json_buffer, orient="records")
            json_b64 = base64.b64encode(json_buffer.getvalue().encode()).decode()
            href = f'<a href="data:application/json;base64,{json_b64}" download="metrics.json">📥 Download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Create a visualization section
        st.markdown(
            "<h2 class='sub-header'>Metrics Visualization</h2>", unsafe_allow_html=True
        )

        # Add tabs for single metric vs multi-metric visualization
        viz_tab1, viz_tab2 = st.tabs(["Single Metric", "Multi-Metric Comparison"])

        # Filter metrics to only show those that are in the dataframe
        available_metrics = {k: v for k, v in METRICS_MAP.items() if v in df.columns}

        with viz_tab1:
            if available_metrics:
                with st.container():
                    st.markdown('<div class="metric-selector">', unsafe_allow_html=True)
                    user_metric_input = st.radio(
                        "💹 Choose metric to visualize:",
                        list(available_metrics.keys()),
                        horizontal=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    with st.spinner("Generating chart..."):
                        chart = plot_line(workdir, available_metrics[user_metric_input])
                        st.plotly_chart(
                            chart, use_container_width=True, theme="streamlit"
                        )

                        # Add chart export options
                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            # Export chart as PNG
                            try:
                                chart_fig = chart
                                img_bytes = chart_fig.to_image(
                                    format="png", engine="kaleido"
                                )
                                b64 = base64.b64encode(img_bytes).decode()
                                href = f'<a href="data:image/png;base64,{b64}" download="chart_{available_metrics[user_metric_input]}.png">📥 Download Chart as PNG</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"Could not generate PNG export: {str(e)}")

                        with export_col2:
                            # Export as PDF report
                            if st.button("📑 Generate PDF Report"):
                                try:
                                    with st.spinner("Generating PDF report..."):
                                        metric_col = available_metrics[
                                            user_metric_input
                                        ]
                                        pdf_buffer = generate_pdf_report(
                                            df, chart, metric_col, workdir
                                        )
                                        pdf_b64 = base64.b64encode(
                                            pdf_buffer.read()
                                        ).decode()
                                        href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="metrics_report_{metric_col}.pdf">📥 Download PDF Report</a>'
                                        st.markdown(href, unsafe_allow_html=True)
                                except Exception as e:
                                    st.warning(
                                        f"Could not generate PDF report: {str(e)}"
                                    )

                    # Add summary metrics
                    if len(df) > 1:
                        st.markdown(
                            "<h3 class='sub-header'>Summary Statistics</h3>",
                            unsafe_allow_html=True,
                        )

                        metric_col = available_metrics[user_metric_input]

                        # Create a 4-column layout for key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(
                                f"""
                            <div class="metric-card">
                                <h4>Initial Value</h4>
                                <h2>{df[metric_col].iloc[0]:.4f}</h2>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col2:
                            st.markdown(
                                f"""
                            <div class="metric-card">
                                <h4>Current Value</h4>
                                <h2>{df[metric_col].iloc[-1]:.4f}</h2>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col3:
                            improvement = (
                                df[metric_col].iloc[-1] - df[metric_col].iloc[0]
                            )
                            color = "#4CAF50" if improvement > 0 else "#F44336"
                            st.markdown(
                                f"""
                            <div class="metric-card">
                                <h4>Absolute Improvement</h4>
                                <h2 style="color: {color};">{improvement:.4f}</h2>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with col4:
                            if df[metric_col].iloc[0] != 0:
                                rel_improvement = (
                                    df[metric_col].iloc[-1] / df[metric_col].iloc[0] - 1
                                ) * 100
                                color = "#4CAF50" if rel_improvement > 0 else "#F44336"
                                st.markdown(
                                    f"""
                                <div class="metric-card">
                                    <h4>Relative Improvement</h4>
                                    <h2 style="color: {color};">{rel_improvement:.2f}%</h2>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"""
                                <div class="metric-card">
                                    <h4>Relative Improvement</h4>
                                    <h2>N/A</h2>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
            else:
                st.markdown(
                    """
                <div class="empty-state">
                    <h3>No metrics available for visualization</h3>
                    <p>The current experiment doesn't have any recognized metrics to display.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        with viz_tab2:
            if available_metrics:
                import plotly.graph_objects as go

                st.markdown('<div class="metric-selector">', unsafe_allow_html=True)
                selected_metrics = st.multiselect(
                    "Select metrics to compare:",
                    options=list(available_metrics.keys()),
                    default=(
                        [list(available_metrics.keys())[0]]
                        if available_metrics
                        else None
                    ),
                    max_selections=5,
                    help="Select up to 5 metrics to compare on a single chart",
                )

                # Add normalization option
                normalize = st.checkbox(
                    "Normalize metrics (0-1 scale)",
                    value=True,
                    help="Normalize all metrics to a 0-1 scale for easier comparison",
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if selected_metrics:
                    # Create a multi-line plot using Plotly
                    with st.spinner("Generating comparison chart..."):
                        fig = go.Figure()

                        # Create a copy of the DataFrame for normalization
                        plot_df = df.copy()

                        for metric_name in selected_metrics:
                            metric_col = available_metrics[metric_name]

                            y_values = plot_df[metric_col].values

                            # Normalize if chosen
                            if normalize and len(y_values) > 0:
                                min_val = min(y_values)
                                max_val = max(y_values)
                                if max_val > min_val:
                                    y_values = [
                                        (y - min_val) / (max_val - min_val)
                                        for y in y_values
                                    ]
                                    legend_name = f"{metric_name} (norm)"
                                else:
                                    legend_name = metric_name
                            else:
                                legend_name = metric_name

                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(len(plot_df))),
                                    y=y_values,
                                    mode="lines+markers",
                                    name=legend_name,
                                    hovertemplate=f"{metric_name}: %{{y:.4f}}<extra></extra>",
                                )
                            )

                        # Update layout
                        fig.update_layout(
                            title="Multi-Metric Comparison",
                            xaxis_title="Iteration",
                            yaxis_title=(
                                "Metric Value"
                                if not normalize
                                else "Normalized Value (0-1)"
                            ),
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                            ),
                            height=500,
                        )

                        # Add vertical lines at each iteration
                        for i in range(1, len(plot_df)):
                            fig.add_shape(
                                type="line",
                                x0=i,
                                y0=0,
                                x1=i,
                                y1=1,
                                line=dict(color="LightGrey", width=1, dash="dot"),
                                xref="x",
                                yref="paper",
                            )

                        st.plotly_chart(
                            fig, use_container_width=True, theme="streamlit"
                        )

                        # Add export option for multi-metric chart
                        try:
                            img_bytes = fig.to_image(format="png", engine="kaleido")
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/png;base64,{b64}" download="multi_metric_comparison.png">📥 Download Comparison Chart as PNG</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not generate PNG export: {str(e)}")

                    # Show metrics details side by side
                    if len(selected_metrics) > 1 and len(df) > 1:
                        # Just display a simple message instead of the comparison table
                        st.info(
                            "Multiple metrics selected. You can compare their trends in the chart above."
                        )
                    else:
                        st.info("Please select at least one metric to visualize")
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                        <h3>No metrics available for visualization</h3>
                        <p>The current experiment doesn't have any recognized metrics to display.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
        <div class="empty-state">
            <h3>No results found</h3>
            <p>There are no metrics available at the moment. Please wait for the experiment to generate some results.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
