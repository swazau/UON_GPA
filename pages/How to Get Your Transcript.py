import streamlit as st
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="How to Get Your Transcript - UON Transcript Analyser",
    page_icon=os.path.join("screenshots", "logo-wamgpt.svg"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title("How to Get Your Academic Transcript")

st.markdown("""
## University of Newcastle Transcript Access Guide

### Step 1: Log in to myUni
1. Go to [myUni](https://myuni.newcastle.edu.au)
2. Log in with your student credentials
3. Click myHub

### Step 2: Request Unofficial Transcript
1. In myHub, click on "Academic Records"
2. Select "Unofficial Transcript"
3. Click "Request"

### Step 3: Download Your Transcript
1. Once processed (usually within minutes), you'll receive a notification
2. Go to "View Requested Documents" 
3. Download your PDF transcript
4. Save it to your computer for uploading to the Transcript Analyser

### Important Notes
- The transcript should be in PDF format
- The Transcript Analyser can use both unofficial and official transcripts

### Need Further Assistance?
- Visit the [Student Services Hub](https://askuon.newcastle.edu.au) for help with obtaining your transcript
- For technical issues with the Transcript Analyser, [report them on GitHub](https://github.com/swazau/UON_GPA/issues)
""")

# Add a button to return to the main page
if st.button("Return to Transcript Analyser"):
    st.switch_page("Dashboard.py")  # This will navigate back to the main dashboard

# Add a sidebar with additional information (copied from main dashboard for consistency)
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