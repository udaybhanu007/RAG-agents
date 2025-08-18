import streamlit as st
import sys
import os
import time
import asyncio
from typing import Optional

# Add path to your workflow - since we're in the agents folder, adjust the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert at the beginning to ensure our modules are found first
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Try to import agentic system first
    from updated_agents.agentic_main import AgenticRAGApplication
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False

try:
    # Fallback to original system
    from agents.multi_agent_rag_workflow import MultiAgentRAGWorkflow
    from core.security_middleware import SecurityViolationError
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    ORIGINAL_AVAILABLE = False
    if not AGENTIC_AVAILABLE:
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
    
    /* Hide Streamlit default elements and reduce top spacing */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Remove default top padding from Streamlit */
    .main .block-container {
        padding-top: 1rem !important;
    }
    
    /* Remove top margin from first element */
    .main .block-container > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Main app container */
    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content area - compact design with minimal top padding */
    .main .block-container {
        padding: calc(var(--spacing-unit) * 0.5) calc(var(--spacing-unit) * 1.5);
        max-width: 800px;
        margin: 0 auto;
        padding-top: calc(var(--spacing-unit) * 0.5) !important;
    }
    
    /* Header styling - compact */
    .app-header {
        background: var(--background-card);
        padding: calc(var(--spacing-unit) * 1.5);
        border-radius: var(--border-radius-sm);
        box-shadow: var(--shadow-sm);
        margin-bottom: calc(var(--spacing-unit) * 1.5);
        border: 1px solid var(--border-color);
    }
    
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: calc(var(--spacing-unit) * 0.5) 0 0 0;
        font-weight: 500;
    }
    
    /* Card styling - compact */
    .custom-card {
        background: var(--background-card);
        padding: calc(var(--spacing-unit) * 1.5);
        border-radius: var(--border-radius-sm);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        margin-bottom: calc(var(--spacing-unit) * 1);
        transition: all 0.2s ease-in-out;
    }
    
    .custom-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* Status indicator - compact */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: calc(var(--spacing-unit) * 0.75) calc(var(--spacing-unit) * 1.5);
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: calc(var(--spacing-unit) * 1);
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
    
    /* Text Input styling - compact white textbox */
    .stTextInput input {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: var(--border-radius-sm) !important;
        padding: calc(var(--spacing-unit) * 1.5) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease-in-out !important;
        color: #1e293b !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        height: 2.5rem !important;
        box-sizing: border-box !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1), 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        outline: none !important;
        background: #ffffff !important;
    }
    
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    
    /* Text Input Label styling */
    .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: calc(var(--spacing-unit) * 1) !important;
    }
    
    /* Button alignment fix - align with text input */
    .element-container:has(.stButton) {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-end !important;
        height: 100% !important;
    }
    
    /* Ensure buttons align with input field bottom */
    div[data-testid="column"]:has(.stButton) {
        display: flex !important;
        align-items: flex-end !important;
        padding-top: 2rem !important; /* Account for label space */
    }
    
    /* Button styling - compact */
    .stButton button {
        background: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius-sm) !important;
        padding: calc(var(--spacing-unit) * 1.25) calc(var(--spacing-unit) * 2.5) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: var(--shadow-sm) !important;
        height: 2.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
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
    
    /* Example queries styling - compact */
    .example-queries {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-sm);
        padding: calc(var(--spacing-unit) * 1);
        margin-bottom: calc(var(--spacing-unit) * 1);
    }
    
    .example-item {
        background: var(--background-light);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-sm);
        padding: calc(var(--spacing-unit) * 0.75);
        margin: calc(var(--spacing-unit) * 0.25) 0;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        font-size: 0.85rem;
        color: var(--text-primary);
        line-height: 1.3;
    }
    
    .example-item:hover {
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
        transform: translateX(calc(var(--spacing-unit) * 1));
    }
    
    /* Result styling - reduced height */
    .result-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-sm);
        padding: calc(var(--spacing-unit) * 1);
        margin-top: calc(var(--spacing-unit) * 1);
        box-shadow: var(--shadow-sm);
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
    if 'agentic_app' not in st.session_state:
        st.session_state.agentic_app = None
    if 'workflow_ready' not in st.session_state:
        st.session_state.workflow_ready = False
    if 'agentic_ready' not in st.session_state:
        st.session_state.agentic_ready = False
    if 'use_agentic' not in st.session_state:
        st.session_state.use_agentic = AGENTIC_AVAILABLE
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""

# Initialize the workflow
def initialize_workflow():
    """Initialize the Multi-Agent RAG Workflow silently without UI messages"""
    try:
        # Silent initialization without any UI progress messages
        workflow = MultiAgentRAGWorkflow()
        return workflow, True, None
        
    except Exception as e:
        return None, False, str(e)

# Initialize the agentic application
async def initialize_agentic_app():
    """Initialize the Agentic RAG Application"""
    try:
        app = AgenticRAGApplication()
        success = await app.initialize()
        return app, success, None
    except Exception as e:
        return None, False, str(e)

def display_status_indicator(ready: bool, agentic_ready: bool = False, error: Optional[str] = None):
    """Display the current system status"""
    if AGENTIC_AVAILABLE and st.session_state.use_agentic:
        if error:
            st.markdown(f"""
            <div class="status-indicator status-error">
                ❌ Agentic System Error: {error}
            </div>
            """, unsafe_allow_html=True)
        elif agentic_ready:
            st.markdown("""
            <div class="status-indicator status-ready">
                🤖 Agentic System Ready - Autonomous AI with Learning
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-loading">
                <div class="loading-spinner"></div>
                ⏳ Initializing Agentic System...
            </div>
            """, unsafe_allow_html=True)
    else:
        # Original status display logic (commented out to remove status messages)
        pass

def display_example_queries():
    """Display example queries in compact format"""
    st.markdown("""
    <h3 style="color: var(--text-primary); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600; margin-top: calc(var(--spacing-unit) * 2);">
        💡 Example Queries
    </h3>
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

def process_query(workflow, query: str):
    """Process a query through the workflow with error handling and timing"""
    try:
        start_time = time.time()
        
        if st.session_state.use_agentic and AGENTIC_AVAILABLE and hasattr(workflow, 'process_query'):
            # Display agentic processing status
            with st.spinner("� Processing with autonomous agentic reasoning..."):
                # Use asyncio for agentic processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(workflow.process_query(query))
                finally:
                    loop.close()
            
            # Enhanced result display for agentic system
            st.markdown("""
            <h3 style="color: var(--success-color); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600; margin-top: calc(var(--spacing-unit) * 2);">
                🤖 Agentic Analysis Result
            </h3>
            """, unsafe_allow_html=True)
            
            # Display answer
            answer = result.get('answer', 'No answer provided')
            st.markdown(f"""
            <div style="background: var(--background-card); padding: calc(var(--spacing-unit) * 1.5); border-radius: var(--border-radius-sm); border: 1px solid var(--border-color); border-left: 4px solid #8b5cf6; margin-bottom: calc(var(--spacing-unit) * 1); font-size: 0.95rem; line-height: 1.4;">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Display agentic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                quality_score = result.get('answer_quality', 0)
                st.metric("Quality Score", f"{quality_score:.2f}")
            with col2:
                execution_time = result.get('execution_time', 0)
                st.metric("Response Time", f"{execution_time:.2f}s")
            with col3:
                sources_used = len(result.get('sources_used', []))
                st.metric("Sources Used", sources_used)
            with col4:
                trace_id = result.get('trace_id', 'N/A')
                st.metric("Trace ID", trace_id[-8:] if trace_id != 'N/A' else 'N/A')
            
            # Show learning indicator if applicable
            if result.get('learning_applied'):
                st.markdown("""
                <div style="background: linear-gradient(135deg, #06d6a0, #10b981); color: white; padding: calc(var(--spacing-unit) * 1); border-radius: var(--border-radius-sm); font-size: 0.9rem; font-weight: 500; margin: calc(var(--spacing-unit) * 1) 0;">
                    🎓 Learning Applied: System adapted based on execution patterns
                </div>
                """, unsafe_allow_html=True)
            
        else:
            # Original workflow processing
            with st.spinner("🧠 Processing your query through the multi-agent RAG workflow..."):
                result = workflow.run(query)
            
            # Original result display
            st.markdown("""
            <h3 style="color: var(--success-color); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600; margin-top: calc(var(--spacing-unit) * 2);">
                ✅ Query Result
            </h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: var(--background-card); padding: calc(var(--spacing-unit) * 1.5); border-radius: var(--border-radius-sm); border: 1px solid var(--border-color); border-left: 4px solid var(--success-color); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 0.95rem; line-height: 1.4;">
                {result}
            </div>
            """, unsafe_allow_html=True)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Store in history
        st.session_state.query_history.append({
            'query': query,
            'result': result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'response_time': response_time,
            'system_type': 'agentic' if st.session_state.use_agentic else 'original'
        })
        
        return True
        
    except SecurityViolationError as e:
        st.markdown(f"""
        <div class="result-container result-error">
            <h3 style="color: var(--error-color); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600;">
                🔒 Security Violation
            </h3>
            <p style="color: var(--text-primary); margin: 0; font-size: 0.95rem; line-height: 1.4;">
                {str(e)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        return False
        
    except Exception as e:
        st.markdown(f"""
        <div class="result-container result-error">
            <h3 style="color: var(--error-color); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600;">
                ❌ Processing Error
            </h3>
            <p style="color: var(--text-primary); margin: 0; font-size: 0.95rem; line-height: 1.4;">
                {str(e)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        return False

def main():
    """Main Streamlit application"""
    # Page configuration
    if AGENTIC_AVAILABLE:
        st.set_page_config(
            page_title="Agentic Multi-Agent RAG System",
            page_icon="�",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    else:
        st.set_page_config(
            page_title="Multi-Agent RAG System",
            page_icon="�🧠",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # App header with system selection
    if AGENTIC_AVAILABLE:
        st.markdown("""
        <div class="app-header">
            <h1 class="app-title">� Enhanced Multi-Agent RAG System</h1>
            <p class="app-subtitle">Choose between Original and Agentic AI with Autonomous Reasoning & Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System selection
        col1, col2 = st.columns([3, 1])
        with col1:
            system_choice = st.radio(
                "Select System:",
                ["🤖 Agentic System (Autonomous + Learning)", "�🧠 Original System (Static)"],
                index=0 if st.session_state.use_agentic else 1,
                horizontal=True
            )
            st.session_state.use_agentic = "Agentic" in system_choice
        
        with col2:
            if st.button("🔄 Reset Learning", help="Reset all learning data in agentic system"):
                if st.session_state.agentic_app:
                    st.session_state.agentic_app.reset_learning_state()
                    st.success("Learning state reset!")
    else:
        st.markdown("""
        <div class="app-header">
            <h1 class="app-title">🧠 Multi-Agent RAG System</h1>
            <p class="app-subtitle">Intelligent Document Analysis with Advanced AI Agents</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize appropriate workflow
    if st.session_state.use_agentic and AGENTIC_AVAILABLE:
        # Initialize agentic system
        if 'agentic_app' not in st.session_state or st.session_state.agentic_app is None:
            with st.spinner("🤖 Initializing Agentic System..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    app, agentic_ready, error = loop.run_until_complete(initialize_agentic_app())
                    st.session_state.agentic_app = app
                    st.session_state.agentic_ready = agentic_ready
                    st.session_state.agentic_error = error
                finally:
                    loop.close()
        
        workflow = st.session_state.agentic_app
        workflow_ready = st.session_state.agentic_ready
        error = st.session_state.get('agentic_error')
        
    else:
        # Initialize original system
        if 'workflow' not in st.session_state or st.session_state.workflow is None:
            workflow, workflow_ready, error = initialize_workflow()
            st.session_state.workflow = workflow
            st.session_state.workflow_ready = workflow_ready
            st.session_state.workflow_error = error
        else:
            workflow = st.session_state.workflow
            workflow_ready = st.session_state.workflow_ready
            error = st.session_state.get('workflow_error')
    
    # Display status
    display_status_indicator(workflow_ready, st.session_state.get('agentic_ready', False), error)
    
    # Main interface
    if workflow_ready and workflow:
        # Query input section - compact alignment
        system_name = "Agentic" if st.session_state.use_agentic else "Standard"
        st.markdown(f"### 💬 Ask Your Question ({system_name} Mode)")
        
        # Create container for better alignment
        query_container = st.container()
        
        with query_container:
            # Query input with buttons in the same row
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                query = st.text_input(
                    "Your Query:",
                    value=st.session_state.last_query,
                    placeholder="Enter your medical or data analysis query here...",
                    help="Ask questions about medical imaging, patient data, or request specific analysis",
                    label_visibility="visible"
                )
            
            with col2:
                # Compact spacing for alignment
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                search_clicked = st.button(
                    "🔍 Search",
                    type="primary",
                    use_container_width=True,
                    disabled=not query or not query.strip()
                )
            
            with col3:
                # Same compact spacing for alignment
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
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
        
        # Display additional agentic features if available
        if st.session_state.use_agentic and AGENTIC_AVAILABLE and workflow:
            with st.expander("📊 Agentic System Performance", expanded=False):
                capabilities_report = workflow.get_agentic_capabilities_report()
                workflow_perf = capabilities_report.get('workflow_performance', {})
                recent_perf = workflow_perf.get('recent_performance', {})
                
                if recent_perf:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        success_rate = recent_perf.get('success_rate', 0)
                        st.metric("Success Rate", f"{success_rate:.1%}")
                    with col2:
                        avg_quality = recent_perf.get('avg_answer_quality', 0)
                        st.metric("Avg Quality", f"{avg_quality:.2f}")
                    with col3:
                        avg_time = recent_perf.get('avg_execution_time', 0)
                        st.metric("Avg Time", f"{avg_time:.1f}s")
                else:
                    st.info("No performance data available yet. Process some queries to see metrics.")
        
    else:
        # Error state
        system_name = "Agentic Multi-Agent RAG" if st.session_state.use_agentic else "Multi-Agent RAG"
        st.markdown(f"""
        <div class="custom-card">
            <h3 style="color: var(--error-color); margin-bottom: calc(var(--spacing-unit) * 1); font-size: 1rem; font-weight: 600;">
                ⚠️ {system_name} System Unavailable
            </h3>
            <p style="color: var(--text-secondary); margin: 0; font-size: 0.95rem; line-height: 1.4;">
                The {system_name} Workflow could not be initialized. Please check the system configuration and try again.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Retry Initialization"):
            # Clear session state to force re-initialization
            keys_to_clear = ['workflow', 'workflow_ready', 'workflow_error', 'agentic_app', 'agentic_ready', 'agentic_error']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
