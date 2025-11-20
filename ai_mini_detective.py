import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="🕵️ AI Mini-Detective",
    page_icon="🔍",
    layout="centered"
)

# Initialize session state
if 'investigation_started' not in st.session_state:
    st.session_state.investigation_started = False
    st.session_state.clues = []
    st.session_state.hypotheses = []
    st.session_state.conclusion = ""

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ Please set the GEMINI_API_KEY environment variable in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model with Gemini 2.5 Flash
model = genai.GenerativeModel('gemini-2.5-flash')

def generate_response(prompt: str) -> str:
    """Generate a response using the Gemini model."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def extract_clues(mystery: str) -> str:
    """Extract clues from the mystery description."""
    prompt = f"""You are a Senior Clue Extractor. Extract and list all relevant clues from the following mystery:
    
    {mystery}
    
    Extract and format as bullet points:
    - Key objects mentioned
    - Locations
    - People/suspects
    - Timeline of events
    - Any other relevant information"""
    return generate_response(prompt)

def build_hypotheses(clues: str) -> str:
    """Generate hypotheses based on the extracted clues."""
    prompt = f"""You are a Lead Detective. Based on these clues, generate 1-2 plausible hypotheses:
    
    {clues}
    
    For each hypothesis:
    1. State the theory
    2. List supporting evidence
    3. Identify any gaps in information"""
    return generate_response(prompt)

def evaluate_scenario(hypotheses: str) -> str:
    """Evaluate the hypotheses and provide a conclusion."""
    prompt = f"""You are a Chief Probability Analyst. Evaluate these hypotheses and provide a conclusion:
    
    {hypotheses}
    
    For each hypothesis:
    1. Rate its probability (1-100%)
    2. Explain the reasoning
    
    Finally, provide:
    - The most likely scenario
    - Recommended next steps"""
    return generate_response(prompt)

def process_mystery(mystery: str) -> Dict[str, str]:
    """Process the mystery through all steps and return results."""
    # Step 1: Extract clues
    clues = extract_clues(mystery)
    
    # Step 2: Build hypotheses
    hypotheses = build_hypotheses(clues)
    
    # Step 3: Evaluate scenario
    evaluation = evaluate_scenario(hypotheses)
    
    return {
        'clues': clues,
        'hypotheses': hypotheses,
        'evaluation': evaluation
    }

def main():
    st.title("🕵️ AI Mini-Detective — Micro Investigation Game")
    st.markdown("### Crack the case with AI-powered investigation!")
    
    # Initialize session state
    if 'investigation_started' not in st.session_state:
        st.session_state.investigation_started = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    with st.form("mystery_form"):
        mystery_text = st.text_area(
            "Describe your mystery here...", 
            placeholder="Example: I lost my USB drive after coming from class.",
            height=150
        )
        
        submitted = st.form_submit_button("Investigate 🔍", type="primary")
        
        if submitted and mystery_text.strip():
            with st.spinner("🔍 Investigating your mystery..."):
                try:
                    st.session_state.results = process_mystery(mystery_text)
                    st.session_state.investigation_started = True
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during the investigation: {str(e)}")
                    st.session_state.investigation_started = False

    if st.session_state.investigation_started and st.session_state.results:
        results = st.session_state.results
        
        st.divider()
        st.subheader("🔍 Extracted Clues")
        st.markdown(results['clues'])
        
        st.divider()
        st.subheader("🧩 Possible Scenarios")
        st.markdown(results['hypotheses'])
        
        st.divider()
        st.subheader("🎯 Analysis & Conclusion")
        st.success(results['evaluation'])
        
        st.balloons()
        st.markdown("---")
        st.markdown("### 🕵️‍♂️ Case Status: *Investigation Complete*")
        
        if st.button("Start New Investigation", type="primary"):
            st.session_state.results = None
            st.rerun()

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        st.error("⚠️ Please set the GEMINI_API_KEY environment variable in your .env file.")
        st.info("Create or update your .env file with: GEMINI_API_KEY=your-api-key-here")
    else:
        main()
