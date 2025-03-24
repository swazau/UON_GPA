import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from gpa import GPAVisualiser
from wam import WAMCalculator
from transcript_processor import process_transcript

# Set page title
st.title("Transcript Analysis Dashboard")

# File uploader for PDF transcript
uploaded_file = st.file_uploader("Upload your PDF transcript", type="pdf")

if uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner("Processing your transcript..."):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Process the PDF to generate a CSV
        csv_path = process_transcript(pdf_path)

        if csv_path and os.path.exists(csv_path):
            try:
                # Load the CSV into a DataFrame
                df = pd.read_csv(csv_path)

                # Initialize visualizers
                gpa_vis = GPAVisualiser()
                wam_calc = WAMCalculator()

                # Calculate GPA
                gpa_results = gpa_vis.calculate_university_gpa(df)

                # GPA Visualizations Section
                st.subheader("GPA Visualizations")
                st.write("**Grade Distribution**")
                fig = gpa_vis.plot_grade_distribution(df)
                st.plotly_chart(fig, use_container_width=True)

                st.write("**Mark Distribution**")
                fig = gpa_vis.plot_mark_distribution(df)
                st.plotly_chart(fig, use_container_width=True)


                st.write("**Course Performance**")
                fig = gpa_vis.plot_course_performance(df)
                st.plotly_chart(fig, use_container_width=True)

                st.write("**GPA Trend**")
                fig = gpa_vis.plot_gpa_trend(df)
                st.plotly_chart(fig, use_container_width=True)

                # Calculate Honours WAM (2000+ level courses)
                wam_results = wam_calc.calculate_wam(df, 'level', 2000)

                # WAM Visualizations Section
                st.subheader("WAM Visualizations")
                st.write("**WAM Comparison**")
                fig = wam_calc.plot_wam_comparison(df)
                st.plotly_chart(fig, use_container_width=True)

                st.write("**Mark Distribution with Thresholds**")
                fig = wam_calc.plot_mark_distribution(df)
                st.plotly_chart(fig, use_container_width=True)


                st.write("**WAM Trend**")
                fig = wam_calc.plot_wam_trend(df)
                st.plotly_chart(fig, use_container_width=True)

                st.write("**Honours Thresholds**")
                fig = wam_calc.plot_honours_thresholds(df)
                st.plotly_chart(fig, use_container_width=True)


                # Display Calculation Results
                st.subheader("Calculation Results")
                col9, col10 = st.columns(2)
                with col9:
                    st.write("**GPA**")
                    st.write(f"GPA: {gpa_results['gpa']:.2f}")
                with col10:
                    st.write("**Honours WAM (2000+ Level Courses)**")
                    st.write(f"Raw WAM: {wam_results['raw_wam']:.2f}")
                    st.write(f"Rounded WAM: {wam_results['rounded_wam']}")
                    st.write(f"Honours Class: {wam_results['honours_class']}")

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
            finally:
                # Clean up temporary files
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                if os.path.exists(csv_path):
                    os.unlink(csv_path)
        else:
            st.error("Failed to process the transcript. Please ensure the PDF is in the correct format.")
else:
    st.info("Please upload a PDF transcript to begin.")

# Add a footer
st.markdown("---")
st.write("Built with Streamlit | Data processed from your academic transcript")