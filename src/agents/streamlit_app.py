import streamlit as st
import sys
import os
import time
from typing import Optional

# Add path to your workflow - since we're in the agents folder, adjust the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert at the beginning to ensure our modules are found first
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from agents.multi_agent_rag_workflow import MultiAgentRAGWorkflow
    from core.security_middleware import SecurityViolationError
except ImportError as e:
    st.error(f"❌ Error importing required modules: {e}")
    st.error("Make sure you're running from the correct directory and all dependencies are installed")
    st.stop()

# Custom CSS for professional styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary-color: #64748b;
        --accent-color: #0ea5e9;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-light: #f8fafc;
        --background-card: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-radius: 12px;
        --border-radius-sm: 8px;
        --spacing-unit: 8px;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main app container */
    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        padding: calc(var(--spacing-unit) * 4) calc(var(--spacing-unit) * 2);
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .app-header {
        background: var(--background-card);
        padding: calc(var(--spacing-unit) * 3);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        margin-bottom: calc(var(--spacing-unit) * 3);
        border: 1px solid var(--border-color);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .app-subtitle {
        font-size: 1.125rem;
        color: var(--text-secondary);
        margin: calc(var(--spacing-unit) * 1) 0 0 0;
        font-weight: 500;
    }
    
    /* Card styling */
    .custom-card {
        background: var(--background-card);
        padding: calc(var(--spacing-unit) * 3);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        margin-bottom: calc(var(--spacing-unit) * 2);
        transition: all 0.2s ease-in-out;
    }
    
    .custom-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-1px);
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: calc(var(--spacing-unit) * 1) calc(var(--spacing-unit) * 2);
        border-radius: 999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: calc(var(--spacing-unit) * 2);
    }
    
    .status-ready {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-loading {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Textarea styling */
    .stTextArea textarea {
        border: 2px solid var(--border-color) !important;
        border-radius: var(--border-radius-sm) !important;
        padding: calc(var(--spacing-unit) * 2) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        transition: all 0.2s ease-in-out !important;
        background: var(--background-card) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        outline: none !important;
    }
    
    /* Button styling */
    .stButton button {
        background: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius-sm) !important;
        padding: calc(var(--spacing-unit) * 1.5) calc(var(--spacing-unit) * 3) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton button:hover {
        background: var(--primary-hover) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary button */
    .secondary-button button {
        background: var(--background-card) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    .secondary-button button:hover {
        background: var(--background-light) !important;
        border-color: var(--secondary-color) !important;
    }
    
    /* Example queries styling - compact version */
    .example-queries {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: calc(var(--spacing-unit) * 1);
        margin-bottom: calc(var(--spacing-unit) * 1);
    }
    
    .example-item {
        background: var(--background-light);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-sm);
        padding: calc(var(--spacing-unit) * 1);
        margin: calc(var(--spacing-unit) * 0.5) 0;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        font-size: 0.8rem;
        color: var(--text-primary);
    }
    
    .example-item:hover {
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
        transform: translateX(calc(var(--spacing-unit) * 1));
    }
    
    /* Result styling */
    .result-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: calc(var(--spacing-unit) * 3);
        margin-top: calc(var(--spacing-unit) * 2);
        box-shadow: var(--shadow-md);
    }
    
    .result-success {
        border-left: 4px solid var(--success-color);
    }
    
    .result-error {
        border-left: 4px solid var(--error-color);
        background: rgba(239, 68, 68, 0.05);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
        margin-right: calc(var(--spacing-unit) * 1);
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Metrics styling */
    .stMetric {
        background: var(--background-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius-sm) !important;
        padding: calc(var(--spacing-unit) * 2) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: var(--border-radius-sm) !important;
        border: none !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Success alert */
    .stAlert[data-baseweb="notification"] div:first-child {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--success-color) !important;
    }
    
    /* Error alert */
    .stAlert[data-baseweb="notification"].stAlert > div {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid var(--error-color) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--background-light) !important;
        border-radius: var(--border-radius-sm) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Hide streamlit logo */
    .css-15zrgzn {display: none;}
    .css-eczf16 {display: none;}
    .css-jn99sy {display: none;}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'workflow_ready' not in st.session_state:
        st.session_state.workflow_ready = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""

# Initialize the workflow
@st.cache_resource
def initialize_workflow():
    """Initialize the Multi-Agent RAG Workflow with caching for performance"""
    try:
        workflow = MultiAgentRAGWorkflow()
        return workflow, True, None
    except Exception as e:
        return None, False, str(e)

def display_status_indicator(ready: bool, error: Optional[str] = None):
    """Display the current system status"""
    if error:
        st.markdown(f"""
        <div class="status-indicator status-error">
            ❌ Error: {error}
        </div>
        """, unsafe_allow_html=True)
    elif ready:
        st.markdown("""
        <div class="status-indicator status-ready">
            ✅ System Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-indicator status-loading">
            <div class="loading-spinner"></div>
            ⏳ Initializing...
        </div>
        """, unsafe_allow_html=True)

def display_example_queries():
    """Display example queries in a compact format"""
    st.markdown("""
    <div class="custom-card example-queries">
        <h3 style="color: var(--text-primary); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600;">
            💡 Example Queries
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Reduced and more concise examples
    examples = [
        "What is NIH Chest X-ray?",
        "Main findings in X-ray analysis",
        "Male patients with effusion",
        "Dataset age relationships"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(
                example, 
                key=f"example_{i}",
                help="Click to use this query",
                use_container_width=True
            ):
                st.session_state.last_query = example
                st.rerun()

def process_query(workflow: MultiAgentRAGWorkflow, query: str):
    """Process a query through the workflow with error handling and timing"""
    try:
        start_time = time.time()
        
        # Display processing status
        with st.spinner("🧠 Processing your query through the multi-agent RAG workflow..."):
            result = workflow.run(query)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Store in history
        st.session_state.query_history.append({
            'query': query,
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'response_time': response_time
        })
        
        # Display success result
        st.markdown("""
        <div class="result-container result-success">
            <h3 style="color: var(--success-color); margin-bottom: calc(var(--spacing-unit) * 2); font-size: 1.25rem; font-weight: 600;">
                ✅ Query Result
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: var(--background-card); padding: calc(var(--spacing-unit) * 2); border-radius: var(--border-radius-sm); border: 1px solid var(--border-color); margin-bottom: calc(var(--spacing-unit) * 2);">
            {result}
        </div>
        """, unsafe_allow_html=True)
        
        return True
        
    except SecurityViolationError as e:
        st.markdown(f"""
        <div class="result-container result-error">
            <h3 style="color: var(--error-color); margin-bottom: calc(var(--spacing-unit) * 2); font-size: 1.25rem; font-weight: 600;">
                🔒 Security Violation
            </h3>
            <p style="color: var(--text-primary); margin: 0;">
                {str(e)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        return False
        
    except Exception as e:
        st.markdown(f"""
        <div class="result-container result-error">
            <h3 style="color: var(--error-color); margin-bottom: calc(var(--spacing-unit) * 2); font-size: 1.25rem; font-weight: 600;">
                ❌ Processing Error
            </h3>
            <p style="color: var(--text-primary); margin: 0;">
                {str(e)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        return False

def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Multi-Agent RAG System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # App header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🧠 Multi-Agent RAG System</h1>
        <p class="app-subtitle">Intelligent Document Analysis with Advanced AI Agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize workflow
    workflow, workflow_ready, error = initialize_workflow()
    st.session_state.workflow = workflow
    st.session_state.workflow_ready = workflow_ready
    
    # Display status
    display_status_indicator(workflow_ready, error)
    
    # Main interface
    if workflow_ready and workflow:
        # Query input section - compact design
        st.markdown("### 💬 Ask Your Question")
        
        # Query input with buttons in the same row
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            query = st.text_input(
                "Enter your query:",
                value=st.session_state.last_query,
                placeholder="Enter your medical or data analysis query here...",
                help="Ask questions about medical imaging, patient data, or request specific analysis",
                label_visibility="collapsed"
            )
        
        with col2:
            search_clicked = st.button(
                "🔍 Search",
                type="primary",
                use_container_width=True,
                disabled=not query or not query.strip()
            )
        
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.last_query = ""
                st.rerun()
        
        # Process query
        if search_clicked and query and query.strip():
            process_query(workflow, query.strip())
            # Clear the query after processing
            st.session_state.last_query = ""
        
        # Example queries
        display_example_queries()
        
    else:
        # Error state
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: var(--error-color); margin-bottom: calc(var(--spacing-unit) * 2);">
                ⚠️ System Unavailable
            </h3>
            <p style="color: var(--text-secondary); margin: 0;">
                The Multi-Agent RAG Workflow could not be initialized. Please check the system configuration and try again.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Retry Initialization"):
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()
