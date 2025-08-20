"""
Simple Agentic Streamlit Application

This follows the exact same pattern as working_main.py for initialization and query processing,
ensuring consistent behavior between CLI and web UI interfaces.

Key alignments with working_main.py:
- Uses EnhancedAgenticRAGApplication() instance
- Uses initialize_system() method 
- Looks for 'final_answer' key in results
- Shows confidence_score, sources, and agentic_indicators
"""

import streamlit as st
import sys
import os

# Add path - same pattern as agents folder
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Initialize session state for workflow (following agents folder pattern)
def init_session_state():
    if 'agentic_app' not in st.session_state:
        st.session_state.agentic_app = None
    if 'agentic_ready' not in st.session_state:
        st.session_state.agentic_ready = False

# Initialize the agentic application (following working_main.py pattern)
def initialize_agentic_workflow():
    """Initialize the Enhanced Agentic RAG Application with real components like working_main.py"""
    try:
        from updated_agents.simple_agentic_app import EnhancedAgenticRAGApplication
        
        # Create application instance like working_main.py
        app = EnhancedAgenticRAGApplication()
        
        # Initialize with system method (uses .env.dev automatically)
        result = app.initialize_system()
        success = result.get("status") == "success"
        return app, success, None
        
    except Exception as e:
        return None, False, str(e)

def main():
    """Main Streamlit application - following agents folder pattern exactly"""
    
    # Page configuration
    st.set_page_config(
        page_title="Simple Agentic RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    init_session_state()
    
    # App header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🤖 Simple Agentic RAG System</h1>
        <p>Autonomous • Learning • Adaptive</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agentic system (silent like agents folder)
    if 'agentic_app' not in st.session_state or st.session_state.agentic_app is None:
        app, agentic_ready, error = initialize_agentic_workflow()
        st.session_state.agentic_app = app
        st.session_state.agentic_ready = agentic_ready
        st.session_state.agentic_error = error
    else:
        app = st.session_state.agentic_app
        agentic_ready = st.session_state.agentic_ready
        error = st.session_state.get('agentic_error')
    
    # Main interface
    if agentic_ready and app:
        # Query input section
        st.subheader("💬 Ask Your Question")
        
        # Query input
        query = st.text_input(
            "Your Query:",
            placeholder="Enter your medical or data analysis query here...",
            help="Ask questions about medical imaging, patient data, or request specific analysis"
        )
        
        # Process query
        if st.button("🔍 Search", type="primary", disabled=not query or not query.strip()):
            if query and query.strip():
                with st.spinner("🤖 Processing with agentic intelligence..."):
                    try:
                        result = app.process_query(query.strip())
                        
                        # Display results
                        if result.get('error'):
                            st.error(f"Processing error: {result.get('final_answer', 'Unknown error')}")
                        else:
                            st.success("✅ Agentic processing complete!")
                            
                            # Main answer (using final_answer key like working_main.py)
                            st.subheader("📝 Answer")
                            final_answer = result.get('final_answer', 'No answer generated')
                            st.write(final_answer)
                    
                    except Exception as e:
                        st.error(f"❌ Processing Error: {str(e)}")
    
    else:
        # Error state
        st.markdown(f"""
        <div style="background: #fee2e2; border: 1px solid #fecaca; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <h3 style="color: #dc2626; margin-bottom: 0.5rem;">⚠️ Agentic System Unavailable</h3>
            <p style="color: #7f1d1d; margin: 0;">
                The Agentic RAG System could not be initialized. Error: {error if error else 'Unknown error'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Retry Initialization"):
            # Clear session state to force re-initialization
            keys_to_clear = ['agentic_app', 'agentic_ready', 'agentic_error']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
