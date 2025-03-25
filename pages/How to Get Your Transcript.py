import streamlit as st
import os

# Page config
st.set_page_config(
    page_title="How to Get Your Transcript - UON Transcript Analyser",
    page_icon=os.path.join("screenshots", "logo-wamgpt.svg"),
    layout="wide"
)

# Display logo in the sidebar
# logo_path = os.path.join("screenshots", "logo-wamgpt.svg")
# if os.path.exists(logo_path):
#     st.sidebar.image(logo_path, width=150)

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
- The Transcript Analyser can use both unoffical and offical transcripts

### Need Further Assistance?
- Visit the [Student Services Hub](https://askuon.newcastle.edu.au) for help with obtaining your transcript
- For technical issues with the Transcript Analyser, [report them on GitHub](https://github.com/swazau/UON_GPA/issues)
""")

# Add a button to return to the main page
if st.button("Return to Transcript Analyser"):
    st.switch_page("Dashboard.py")  # Assuming your main app is named Dashboard.py