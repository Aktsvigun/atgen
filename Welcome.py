import streamlit as st
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Welcome to AL-NLG",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for navigation
if "welcome_state" not in st.session_state:
    st.session_state.welcome_state = {
        "countdown": 30,
        "auto_redirect": True,
    }

# Apply custom CSS styling
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .welcome-header {
        text-align: center;
        padding: 3rem 0;
    }
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E88E5, #6200EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .welcome-subtitle {
        font-size: 1.5rem;
        color: #666;
        max-width: 800px;
        margin: 0 auto 0.5rem auto;
    }
    .feature-section {
        margin: 2rem 0;
    }
    .feature-box {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        border: 1px solid #e0e0e0;
    }
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 16px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #333;
    }
    .feature-description {
        color: #666;
        font-size: 1rem;
    }
    .cta-section {
        background-color: #f5f5f5;
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 3rem 0;
    }
    .cta-button {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        text-decoration: none;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .cta-button:hover {
        background-color: #388E3C;
        box-shadow: 0 6px 10px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .benefits-section {
        margin: 4rem 0;
        background-color: #e8f5e9;
        padding: 2rem;
        border-radius: 15px;
    }
    .quote-section {
        font-style: italic;
        padding: 2rem;
        border-left: 4px solid #1E88E5;
        background-color: #f9f9f9;
        margin: 2rem 0;
    }
    .stApp {
        background-color: #fafafa;
    }
    .footer-section {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid #e0e0e0;
        color: #757575;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header section
st.markdown(
    """
    <div class="welcome-header">
        <h1 class="welcome-title">Active Learning for Natural Language Generation</h1>
        <p class="welcome-subtitle">
            Train better text generation models with fewer labeled examples through intelligent data selection
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature section
st.markdown(
    "<h2 style='text-align: center; margin: 1rem 0 2rem 0;'>Key Features</h2>",
    unsafe_allow_html=True,
)

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">Intelligent Data Selection</h3>
            <p class="feature-description">
                Choose the most informative examples for annotation using state-of-the-art active learning strategies. Reduce annotation costs while maximizing model performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col2:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Comprehensive Metrics</h3>
            <p class="feature-description">
                Track model performance with a wide range of evaluation metrics. Visualize progress across iterations and compare different strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col3:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-icon">🔄</div>
            <h3 class="feature-title">Iterative Improvement</h3>
            <p class="feature-description">
                Continuously improve your models through an iterative active learning loop. See results get better with each round of data selection and training.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

feature_col4, feature_col5, feature_col6 = st.columns(3)

with feature_col4:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">Flexible Model Support</h3>
            <p class="feature-description">
                Work with a variety of language models, from small open-source models to large commercial API-based systems. Use what works best for your use case.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col5:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-icon">👁️</div>
            <h3 class="feature-title">Interactive Visualization</h3>
            <p class="feature-description">
                Explore your data and model outputs through intuitive visualizations. Compare experiments side by side to identify the best approaches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col6:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-icon">⚙️</div>
            <h3 class="feature-title">Customizable Workflows</h3>
            <p class="feature-description">
                Configure every aspect of your active learning pipeline, from data preprocessing to model training parameters. Tailor the system to your specific needs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Benefits section
st.markdown(
    """
    <div class="benefits-section">
        <h2 style="text-align: center; margin-bottom: 2rem;">Why Active Learning?</h2>
        <div style="max-width: 800px; margin: 0 auto;">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                Active Learning is a machine learning paradigm where an algorithm can interactively query a source 
                (such as a human annotator) to label new data points with the desired outputs.
            </p>
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                <strong>For text generation tasks, this means:</strong>
            </p>
            <ul style="font-size: 1.1rem; margin-left: 2rem;">
                <li>Reduced annotation costs by up to 70%</li>
                <li>Better performance with less training data</li>
                <li>Targeted improvement on challenging examples</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quote section
# st.markdown(
#     """
#     <div class="quote-section">
#         <p style="font-size: 1.2rem;">
#             "Active learning represents one of the most promising approaches to building high-performance NLG systems
#             when annotation resources are limited. By focusing human effort on the most informative examples,
#             we can build better models faster and at lower cost."
#         </p>
#         <p style="text-align: right; font-weight: 600; margin-top: 1rem;">— Research in Natural Language Processing</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# CTA section
st.markdown(
    """
    <div class="cta-section">
        <h2 style="font-size: 2rem; margin-bottom: 1rem;">Ready to Get Started?</h2>
        <p style="font-size: 1.2rem; max-width: 700px; margin: 0 auto;">
            Launch the Active Learning platform to start building better natural language generation models today.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add a Streamlit button for navigation instead of HTML link
cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
with cta_col2:
    if st.button(
        "🚀 Enter the Platform", key="enter_platform_button", use_container_width=True
    ):
        st.switch_page("pages/0_Configure_Experiment.py")

# Better redirect approach with buttons for navigation
redirect_container = st.container()

with redirect_container:
    cols = st.columns([3, 1])

    with cols[0]:
        if st.session_state.welcome_state["auto_redirect"]:
            info_message = st.empty()
            info_message.info(
                f"Automatically redirecting to experiment configuration page in {st.session_state.welcome_state['countdown']} seconds..."
            )

    with cols[1]:
        if st.button("Cancel Redirect", key="cancel_redirect"):
            st.session_state.welcome_state["auto_redirect"] = False
            st.rerun()

# Handle the redirect logic with Streamlit's native approach
if st.session_state.welcome_state["auto_redirect"]:
    # Decrement the counter
    st.session_state.welcome_state["countdown"] -= 1

    # Redirect when counter reaches zero
    if st.session_state.welcome_state["countdown"] <= 0:
        st.session_state.welcome_state["auto_redirect"] = False
        st.switch_page("pages/0_Configure_Experiment.py")
    else:
        # Rerun the app without using time.sleep
        st.rerun()

# Footer
st.markdown(
    """
    <div class="footer-section">
        <p>Created with ❤️ for Natural Language Generation</p>
        <p>© 2025 Active Learning for NLG</p>
    </div>
    """,
    unsafe_allow_html=True,
)
