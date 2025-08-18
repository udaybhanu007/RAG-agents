"""
Simple Agentic Streamlit Application

A clean, focused frontend that demonstrates TRUE agentic behavior:
1. Dynamic reasoning and planning
2. Learning from interactions
3. Transparent decision-making process
4. Simple, effective user interface

This follows the roadmap's emphasis on simplicity and effectiveness.
"""

import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simple_agentic_app import enhanced_agentic_app
from core.logging_config import get_logger

# Initialize logger for streamlit app
logger = get_logger("simple_agentic_streamlit")

def initialize_simple_agentic_system():
    """Initialize the simple agentic system"""
    logger.info("streamlit_initialization_started")
    try:
        # Import necessary components
        from core.azure_keyvault_manager import AzureKeyVaultManager
        from langchain_openai import AzureChatOpenAI
        from qdrant_client import QdrantClient
        
        # Initialize components
        keyvault_manager = AzureKeyVaultManager()
        logger.debug("azure_keyvault_manager_initialized")
        
        # Azure OpenAI LLM
        llm = AzureChatOpenAI(
            azure_endpoint=keyvault_manager.get_secret("azure-openai-endpoint"),
            api_key=keyvault_manager.get_secret("azure-openai-api-key"),
            api_version="2024-05-01-preview",
            deployment_name="gpt-4o-mini",
            temperature=0.1
        )
        logger.debug("azure_llm_initialized")
        
        # Simplified vector store connection
        vector_client = QdrantClient(host="localhost", port=6333)
        logger.debug("vector_client_initialized")
        
        # Simplified graph store (placeholder)
        graph_store = {"type": "neo4j", "connection": "bolt://localhost:7687"}
        logger.debug("graph_store_configured")
        
        # Initialize simple agentic app
        result = enhanced_agentic_app.initialize(llm, vector_client, graph_store)
        logger.info("enhanced_agentic_app_initialized", result_status=result.get("status"))
        
        return result["status"] == "success"
        
    except Exception as e:
        logger.error("streamlit_initialization_failed", error=str(e))
        st.error(f"Initialization error: {str(e)}")
        return False

def display_agentic_reasoning(reasoning_plan: Dict[str, Any]):
    """Display the reasoning process in a clean format"""
    if not reasoning_plan:
        return
    
    st.subheader("🧠 Autonomous Reasoning Process")
    
    with st.expander("View Reasoning Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Query Analysis:**")
            st.write(f"• Type: {reasoning_plan.get('query_type', 'Unknown')}")
            st.write(f"• Route Selected: {reasoning_plan.get('selected_route', 'Unknown')}")
            
        with col2:
            st.write("**Decision Making:**")
            learned = reasoning_plan.get('is_learned_decision', False)
            if learned:
                st.write("🎓 **Learned Decision** - Applied past experience")
            else:
                st.write("🤔 **Reasoned Decision** - New reasoning applied")
        
        st.write("**Reasoning:**")
        st.info(reasoning_plan.get('reasoning', 'No reasoning available'))

def display_learning_progress(learning_update: Dict[str, Any]):
    """Display learning progress in a simple format"""
    if not learning_update:
        return
    
    st.subheader("📚 Learning Progress")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quality = learning_update.get('quality_score', 0.0)
        st.metric("Quality Score", f"{quality:.2f}", delta=None)
    
    with col2:
        adaptations = learning_update.get('adaptation_count', 0)
        st.metric("Total Adaptations", adaptations)
    
    with col3:
        patterns = learning_update.get('total_patterns', 0)
        st.metric("Learned Patterns", patterns)

def display_agentic_indicators(indicators: Dict[str, Any]):
    """Display clear indicators of agentic behavior"""
    if not indicators:
        return
    
    st.subheader("🤖 Agentic Capabilities Active")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if indicators.get('autonomous_reasoning', False):
            st.success("✅ Autonomous Reasoning")
        else:
            st.warning("⏳ Reasoning Inactive")
    
    with col2:
        if indicators.get('learning_applied', False):
            st.success("✅ Learning Applied")
        else:
            st.info("💡 Ready to Learn")
    
    with col3:
        if indicators.get('adaptive_behavior', False):
            st.success("✅ Adaptive Behavior")
        else:
            st.warning("⚙️ Static Behavior")

def main():
    """Main Streamlit application"""
    
    # Configure page
    st.set_page_config(
        page_title="Simple Agentic RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for clean styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agentic-indicator {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Simple Agentic RAG System</h1>
        <p>Autonomous • Learning • Adaptive</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for system status
    with st.sidebar:
        st.header("📊 System Status")
        
        # Initialize system
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize Agentic System", type="primary"):
                with st.spinner("Initializing agentic capabilities..."):
                    st.session_state.system_initialized = initialize_simple_agentic_system()
                
                if st.session_state.system_initialized:
                    st.success("✅ System Ready!")
                    st.rerun()
                else:
                    st.error("❌ Initialization Failed")
        
        else:
            st.success("✅ System Active")
            
            # System status
            try:
                status = enhanced_agentic_app.get_system_status()
                
                st.subheader("🎯 Capabilities")
                capabilities = status.get('agentic_capabilities', {})
                
                for capability, active in capabilities.items():
                    icon = "✅" if active else "❌"
                    readable_name = capability.replace('_', ' ').title()
                    st.write(f"{icon} {readable_name}")
                
                # Learning insights
                st.subheader("📚 Learning Status")
                insights = enhanced_agentic_app.get_learning_insights()
                
                st.metric("Adaptations", insights.get('total_adaptations', 0))
                st.metric("Patterns Learned", insights.get('learned_patterns', 0))
                
                # Reset learning button
                if st.button("🔄 Reset Learning"):
                    enhanced_agentic_app.reset_learning()
                    st.success("Learning data reset!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Status error: {str(e)}")
    
    # Main interface
    if not st.session_state.system_initialized:
        st.info("👈 Please initialize the agentic system using the sidebar.")
        return
    
    # Query input
    st.subheader("💬 Ask Your Question")
    
    # Example queries
    with st.expander("💡 Try These Examples"):
        examples = [
            "Compare pneumonia vs COVID-19 patterns in chest X-rays",
            "What are the key diagnostic features for identifying pneumonia?",
            "Analyze the relationship between patient age and diagnosis accuracy",
            "Find patterns in chest X-ray abnormalities across different demographics"
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{examples.index(example)}"):
                st.session_state.current_query = example
    
    # Query input
    query = st.text_area(
        "Enter your medical/healthcare question:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="Ask about medical conditions, diagnostic patterns, or healthcare data analysis..."
    )
    
    # Process query
    if st.button("🔍 Process with Agentic Intelligence", type="primary", disabled=not query.strip()):
        if query.strip():
            
            # Processing animation
            with st.spinner("🤖 Agentic system is reasoning and learning..."):
                
                # Process query
                result = enhanced_agentic_app.process_query(query)
                
                # Check for errors
                if result.get('error'):
                    st.error(f"Processing error: {result.get('answer', 'Unknown error')}")
                    return
                
                # Display results
                st.success("✅ Agentic processing complete!")
                
                # Main answer
                st.subheader("📝 Answer")
                st.write(result.get('answer', 'No answer generated'))
                
                # Agentic indicators
                agentic_indicators = result.get('agentic_indicators', {})
                display_agentic_indicators(agentic_indicators)
                
                # Reasoning process
                reasoning_plan = result.get('reasoning_plan', {})
                display_agentic_reasoning(reasoning_plan)
                
                # Learning progress
                learning_update = result.get('learning_update', {})
                display_learning_progress(learning_update)
                
                # Execution metrics
                metrics = result.get('execution_metrics', {})
                if metrics:
                    st.subheader("⚡ Performance Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        exec_time = metrics.get('execution_time', 0)
                        st.metric("Execution Time", f"{exec_time:.2f}s")
                    
                    with col2:
                        exec_count = metrics.get('execution_count', 0)
                        st.metric("Total Queries", exec_count)
                
                # Sources
                sources = result.get('sources', [])
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources[:5], 1):
                            st.write(f"**Source {i}:** {source}")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    ### 🤖 About This Agentic System
    
    This system demonstrates **TRUE agentic behavior** through:
    
    - **🧠 Autonomous Reasoning**: Dynamic route selection based on query analysis
    - **📚 Continuous Learning**: Adapts strategies based on past performance
    - **🎯 Adaptive Decisions**: Self-optimizing behavior without human intervention
    - **🔍 Transparent Process**: Complete visibility into AI reasoning
    
    The system **learns and improves** with each interaction, becoming more intelligent over time.
    """)

if __name__ == "__main__":
    main()
