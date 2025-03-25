import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import logging
from gpa import GPAVisualiser
from wam import WAMCalculator
from transcript_processor import process_transcript

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="UON Transcript Analyser",
    page_icon=os.path.join("screenshots", "logo-wamgpt.svg"),  # Using your custom SVG for favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display your custom SVG logo from the screenshots folder
# logo_path = os.path.join("screenshots", "logo-wamgpt.svg")
# if os.path.exists(logo_path):
#     st.image(logo_path, width=200)
# else:
#     logger.warning(f"Logo not found at path: {logo_path}")

# Set page title and description
st.title("University of Newcastle Transcript Analysis Dashboard")
st.markdown("""
This tool helps University of Newcastle students analyse their academic transcripts.
Upload your PDF transcript to visualise your GPA, WAM, and course performance metrics.
""")

# Add a session state for tracking processing status
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# File uploader
uploaded_file = st.file_uploader(
    "Upload your PDF transcript",
    type="pdf",
    help="Only University of Newcastle PDF transcripts are supported"
)


def process_and_analyse():
    """Process the uploaded transcript and display results"""
    st.session_state.processing_complete = False
    st.session_state.error_message = None

    try:
        # Create a secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())  # Use getvalue() instead of read()
            pdf_path = tmp_file.name

        logger.info(f"Processing PDF at temporary path: {pdf_path}")

        # Process the PDF to generate a CSV
        try:
            csv_path = process_transcript(pdf_path)
        except Exception as e:
            logger.error(f"Transcript processing error: {str(e)}")
            st.session_state.error_message = "Failed to process transcript. Please ensure you're uploading an UON transcript."
            return

        if not csv_path or not os.path.exists(csv_path):
            logger.error("CSV path not returned or file doesn't exist")
            st.session_state.error_message = "Failed to process the transcript data. Please ensure the PDF is in the correct format."
            return

        # Load the CSV into a DataFrame
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.error("Empty dataframe returned from CSV")
                st.session_state.error_message = "No data could be extracted from the transcript. Please ensure it's a valid UoN transcript."
                return

            # Skip strict column validation to ensure compatibility with existing data
            logger.info(f"DataFrame columns: {list(df.columns)}")

            # Initialise visualisers
            gpa_vis = GPAVisualiser()
            wam_calc = WAMCalculator()

            # Calculate metrics
            gpa_results = gpa_vis.calculate_university_gpa(df)
            wam_results = wam_calc.calculate_wam(df, 'level', 2000)

            # Store in session state
            st.session_state.df = df
            st.session_state.gpa_vis = gpa_vis
            st.session_state.wam_calc = wam_calc
            st.session_state.gpa_results = gpa_results
            st.session_state.wam_results = wam_results
            st.session_state.processing_complete = True

        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            st.session_state.error_message = f"Error analysing transcript data: {str(e)}"
            return

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.session_state.error_message = "An unexpected error occurred. Please try again."
        return
    finally:
        # Clean up temporary files
        try:
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.unlink(pdf_path)
            if 'csv_path' in locals() and os.path.exists(csv_path):
                os.unlink(csv_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")


if uploaded_file is not None:
    # Process automatically when file is uploaded (no button needed)
    with st.spinner("Processing your transcript..."):
        process_and_analyse()

# Display error if any
if st.session_state.get('error_message'):
    st.error(st.session_state.error_message)

# Display results if processing is complete
if st.session_state.get('processing_complete', False):
    df = st.session_state.df
    gpa_results = st.session_state.gpa_results
    wam_results = st.session_state.wam_results

    # Create tabs for organisation
    tab1, tab2, tab3 = st.tabs(["GPA Analysis", "WAM Analysis", "Summary Results"])

    with tab1:
        st.header("GPA Visualisations")

        st.subheader("Grade Distribution")
        fig = st.session_state.gpa_vis.plot_grade_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mark Distribution")
        fig = st.session_state.gpa_vis.plot_mark_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Course Performance")
        fig = st.session_state.gpa_vis.plot_course_performance(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("GPA Trend")
        fig = st.session_state.gpa_vis.plot_gpa_trend(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("WAM Visualisations")

        st.subheader("WAM Comparison")
        fig = st.session_state.wam_calc.plot_wam_comparison(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mark Distribution with Thresholds")
        fig = st.session_state.wam_calc.plot_mark_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("WAM Trend")
        fig = st.session_state.wam_calc.plot_wam_trend(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Honours Thresholds")
        fig = st.session_state.wam_calc.plot_honours_thresholds(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Summary Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("GPA", f"{gpa_results['gpa']:.2f}")

            # Add more context about GPA
            st.info("""
            **GPA Scale:** 0-7
            **UoN Equivalent:** 
            - 7.0: High Distinction
            - 6.0: Distinction
            - 5.0: Credit
            - 4.0: Pass
            - <4.0: Fail
            """)

        with col2:
            st.metric("Honours WAM", f"{wam_results['rounded_wam']}")
            st.metric("Honours Class", wam_results['honours_class'])

            # Add more context about WAM
            st.info("""
            **Honours Classes:**
            - First Class: 80+
            - Second Class, Division 1: 75-79
            - Second Class, Division 2: 70-74
            - Third Class: 65-69
            """)

        # Download option for results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="transcript_analysis.csv",
            mime="text/csv",
        )
else:
    # Show sample images if no file uploaded
    if not uploaded_file:
        st.info("Please upload a PDF transcript to begin analysis. [How to Get Your Transcript](/How_to_Get_Your_Transcript)")

        # Initialise session state for showing samples if not already set
        if 'show_samples' not in st.session_state:
            st.session_state.show_samples = False

        # Function to toggle sample visibility
        def toggle_samples():
            st.session_state.show_samples = not st.session_state.show_samples

        # Toggle button for samples with the correct label based on current state
        button_label = "Minimise Samples" if st.session_state.show_samples else "View Sample Visualisations"
        st.button(button_label, on_click=toggle_samples)

        # Display samples if the session state is true
        if st.session_state.show_samples:
            st.subheader("Sample Visualisations")

            # Sample images from screenshots folder
            sample_images = [
                "course_performance.png",
                "gpa_trend.png",
                "grade_distribution.png",
                "honours_thresholds.png",
                "mark_distribution.png",
                "wam_comparison.png",
                "wam_mark_distribution.png",
                "wam_trend.png"
            ]

            # Display sample images in two columns
            col1, col2 = st.columns(2)

            for i, img_name in enumerate(sample_images):
                img_path = os.path.join("screenshots", img_name)
                if os.path.exists(img_path):
                    # Alternate between columns
                    if i % 2 == 0:
                        with col1:
                            st.image(img_path, caption=img_name.replace(".png", "").replace("_", " ").title())
                    else:
                        with col2:
                            st.image(img_path, caption=img_name.replace(".png", "").replace("_", " ").title())
                else:
                    st.warning(f"Sample image not found: {img_path}")
# Add a sidebar with additional information
with st.sidebar:
    # Display logo in the sidebar
    logo_path = os.path.join("screenshots", "logo-wamgpt.svg")
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        logger.warning(f"Logo not found at path: {logo_path}")

    st.header("About")
    st.write("""
    This tool helps University of Newcastle students analyse their academic performance
    through their transcript data.
    """)

    st.header("Need Help?")
    st.markdown("[Report an issue on GitHub](https://github.com/swazau/UON_GPA/issues)")

    st.header("Created By")
    st.write("Daniel Ferguson")
    st.markdown(
        '<a href="https://github.com/swazau" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"> GitHub</a>',
        unsafe_allow_html=True)
    st.write("DataCraftsmanAU")
    st.markdown(
        '<a href="https://github.com/DataCraftsmanAU" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"> GitHub</a>',
        unsafe_allow_html=True)

# Add a footer
st.markdown("---")
st.markdown("© 2025 | University of Newcastle Transcript Analyser")
