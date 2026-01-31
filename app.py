"""
AI Witness Integrity Layer - Hackathon Demo
============================================
A Streamlit dashboard for cross-examining an AI surveillance system.
Automatically loads the latest 'fight' detected by seg.py.
"""

import streamlit as st
import json
import ollama
import os
import glob
from PIL import Image
from typing import Dict, List, Any, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "mistral:7b-instruct-q6_K"  # Make sure this matches 'ollama list'
EVIDENCE_DIR = "intersection_outputs"
PLACEHOLDER_IMAGE = "http://placehold.co/600x400/1a1a1a/ffffff?text=WAITING+FOR+EVENTS..."

# =============================================================================
# DATA LOADING (DYNAMIC)
# =============================================================================

def get_latest_evidence() -> Tuple[Dict[str, Any], str, str]:
    """
    Scans the output directory for the newest JSON and Image files.
    Returns: (json_data, image_path, filename_display)
    """
    # 1. Check if directory exists
    if not os.path.exists(EVIDENCE_DIR):
        return {}, None, "Directory not found"

    # 2. Find all JSON files
    json_files = glob.glob(os.path.join(EVIDENCE_DIR, "*.json"))
    
    if not json_files:
        return {}, None, "No evidence found"

    # 3. Sort by modification time (newest first)
    latest_json_path = max(json_files, key=os.path.getctime)
    
    # 4. Find the matching image (same name, but .jpg)
    # seg.py saves them with matching timestamps
    base_name = os.path.splitext(latest_json_path)[0]
    latest_img_path = base_name + ".jpg"

    # 5. Load the JSON data
    try:
        with open(latest_json_path, 'r') as f:
            data = json.load(f)
            
        filename = os.path.basename(latest_json_path)
        return data, latest_img_path, filename
        
    except Exception as e:
        st.error(f"Error loading evidence: {e}")
        return {}, None, "Error"

# =============================================================================
# SYSTEM PROMPT CONSTRUCTION
# =============================================================================

def build_system_message(evidence_log: Dict[str, Any]) -> str:
    """Build the strict AI Witness persona."""
    json_string = json.dumps(evidence_log, indent=2)
    
    system_message = f"""You are an AI Witness. You did NOT see the video. You ONLY know the JSON logs provided.

EVIDENCE LOG:
{json_string}

Your Job: Defend or explain the flag.

RULES:
1. Cite specific numbers (IoU, Confidence, mask_overlap_percent).
2. Admit uncertainty (lighting, occlusion, shadows).
3. If asked about emotions ("Was he angry?"), REFUSE to answer. You only see pixels.
4. Keep responses concise (2-3 sentences max).
5. Only reference data in the Evidence Log.
"""
    return system_message

# =============================================================================
# OLLAMA INTEGRATION
# =============================================================================

def stream_ollama_response(messages: List[Dict[str, str]]) -> str:
    """Stream response from Ollama locally."""
    try:
        message_placeholder = st.empty()
        full_response = ""
        
        # Call Ollama Python library
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            options={"temperature": 0.3}
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        if "connection" in str(e).lower():
            st.info("💡 Run 'ollama serve' in your terminal.")
        return "System Offline."

# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'evidence_log' not in st.session_state:
        st.session_state.evidence_log = {}

    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
        
    if 'current_file' not in st.session_state:
        st.session_state.current_file = "Waiting for scan..."

    if 'system_message' not in st.session_state:
        st.session_state.system_message = ""

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_evidence_panel():
    """Render the visual evidence and metrics."""
    st.header("📹 Evidence Vault")
    
    # 1. Display Image
    st.subheader("Surveillance Frame")
    if st.session_state.current_image and os.path.exists(st.session_state.current_image):
        image = Image.open(st.session_state.current_image)
        st.image(image, caption=f"Source: {st.session_state.current_file}", use_container_width=True)
    else:
        st.image(PLACEHOLDER_IMAGE, use_container_width=True)

    st.markdown("---")

    # 2. Display Metrics (Only if log is loaded)
    log = st.session_state.evidence_log
    if log:
        st.subheader("Forensic Metrics")
        
        # Safely access nested keys
        decision = log.get("decision", "N/A")
        lighting = log.get("metadata", {}).get("lighting_condition", "N/A")
        
        # Handle geometry safely
        geo = log.get("geometry", {})
        iou = geo.get("mask_overlap_iou", 0)
        contact = geo.get("contact_detected", False)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", decision, delta="ALERT" if decision == "FLAGGED" else None)
        col2.metric("Mask IoU", f"{iou}")
        col3.metric("Lighting", lighting)
        
        with st.expander("📊 View Raw JSON Log", expanded=False):
            st.json(log)
    else:
        st.info("No evidence loaded. Run seg.py and click Scan.")

def render_interrogation_panel():
    """Render the chat interface."""
    st.header("⚖️ Witness Interrogation")
    st.caption(f"Connected to: {MODEL_NAME}")

    # Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if prompt := st.chat_input("Cross-examine the AI..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI Response
        with st.chat_message("assistant"):
            # Prepare context
            msgs = [{"role": "system", "content": st.session_state.system_message}]
            msgs.extend(st.session_state.messages)
            
            response = stream_ollama_response(msgs)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="AI Witness", layout="wide", page_icon="⚖️")
    init_session_state()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.title("🎛️ Controls")
        
        if st.button("🔄 SCAN FOR NEW EVIDENCE", type="primary", use_container_width=True):
            # Load new data
            data, img_path, filename = get_latest_evidence()
            
            if data:
                st.session_state.evidence_log = data
                st.session_state.current_image = img_path
                st.session_state.current_file = filename
                # Rebuild brain
                st.session_state.system_message = build_system_message(data)
                # Reset chat on new evidence
                st.session_state.messages = []
                st.success(f"Loaded: {filename}")
            else:
                st.warning(filename) # Displays error message
                
        st.markdown("---")
        st.caption(f"Currently Loaded:\n{st.session_state.current_file}")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN LAYOUT ---
    st.title("⚖️ AI Witness Integrity Layer")
    
    left, right = st.columns([1, 1.2], gap="large")
    
    with left:
        render_evidence_panel()
    
    with right:
        if st.session_state.evidence_log:
            render_interrogation_panel()
        else:
            st.warning("⚠️ Waiting for Evidence. Run 'seg.py' then click SCAN.")

if __name__ == "__main__":
    main()