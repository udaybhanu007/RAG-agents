"""
Simple Agentic Streamlit Application

A clean production-ready web interface for the Enhanced Agentic RAG System.

Key features:
- Uses EnhancedAgenticRAGApplication() instance
- Supports both Key Vault and direct environment configuration
- Clean professional UI
- Efficient query processing and result display
"""

import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Add path - same pattern as agents folder
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Initialize session state for workflow
def init_session_state():
    if 'agentic_app' not in st.session_state:
        st.session_state.agentic_app = None
    if 'agentic_ready' not in st.session_state:
        st.session_state.agentic_ready = False



# Initialize the agentic application
def initialize_agentic_workflow():
    """Initialize the Enhanced Agentic RAG Application"""
    try:
        from updated_agents.simple_agentic_app import EnhancedAgenticRAGApplication
        
        # Create application instance
        app = EnhancedAgenticRAGApplication()
        
        # Initialize system
        result = app.initialize_system()
        
        success = result.get("status") == "success"
        
        return app, success, None
        
    except Exception as e:
        error_msg = f"Initialization error: {str(e)}"
        return None, False, str(e)

def main():
    """Main Streamlit application - following agents folder pattern exactly"""
    
    # Page configuration
    st.set_page_config(
        page_title="Hybrid Agentic RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # App header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🤖 Hybrid Agentic RAG System</h1>
        <p>Autonomous • Learning • Adaptive • LangGraph Orchestrated</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agentic system
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
        
        # Initialize clear counter for forcing text input reset
        if 'clear_counter' not in st.session_state:
            st.session_state.clear_counter = 0
        
        # Simple query input with dynamic key to force clearing
        query = st.text_input(
            "Your Query:",
            placeholder="Enter your medical or data analysis query here...",
            help="Ask questions about medical imaging, patient data, or request specific analysis",
            key=f"query_input_{st.session_state.clear_counter}"
        )
        
        # Button row with Search and Clear
        col1, col2, col3 = st.columns([0.7, 0.7, 5])
        
        with col1:
            search_clicked = st.button("🔍 Search", type="primary", disabled=not query or not query.strip())
        
        with col2:
            if st.button("🗑️ Clear", type="secondary"):
                st.session_state.clear_counter += 1  # Increment to change the key
                st.rerun()  # Rerun to recreate the text input with new key
        
        # Process query immediately when search is clicked
        if search_clicked and query and query.strip():
            with st.spinner("🤖 Processing with agentic intelligence..."):
                try:
                    # Process using LangGraph framework (app now uses LangGraph by default)
                    result = app.process_query(query.strip())
                    
                    # Display results immediately
                    if result.get('error'):
                        error_msg = result.get('final_answer', 'Unknown error')
                        st.error(f"❌ Processing error: {error_msg}")
                    else:
                        st.success("✅ Agentic processing complete!")
                        
                        # Main answer
                        st.subheader("📝 Answer")
                        final_answer = result.get('final_answer', 'No answer generated')
                        st.write(final_answer)
                
                except Exception as e:
                    error_msg = f"Processing Error: {str(e)}"
                    st.error(f"❌ {error_msg}")
    
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retry Initialization"):
                # Clear session state to force re-initialization
                keys_to_clear = ['agentic_app', 'agentic_ready', 'agentic_error']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
