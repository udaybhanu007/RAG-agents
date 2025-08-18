"""
Simple Agentic RAG System Launcher

This script launches the validated agentic RAG system that follows
the architectural roadmap requirements:

1. True agentic behavior with autonomous reasoning
2. Simple, modular approach with maximum code reuse  
3. Minimal complexity and clean implementation
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from core.logging_config import configure_logging, get_logger

def launch_simple_agentic_app():
    """Launch the simple agentic Streamlit application"""
    
    # Configure logging for development mode
    configure_logging(
        log_level="INFO",
        enable_json=False,
        enable_colors=True
    )
    
    logger = get_logger("launcher")
    logger.info("launching_simple_agentic_system")
    
    print("🤖 Launching Simple Agentic RAG System...")
    print("=" * 50)
    print("✅ TRUE AGENTIC BEHAVIOR:")
    print("   • Autonomous reasoning and decision-making")
    print("   • Learning from each interaction")
    print("   • Adaptive strategy optimization")
    print("   • Transparent decision process")
    print()
    print("✅ SIMPLE & MODULAR DESIGN:")
    print("   • Maximum reuse of existing codebase")
    print("   • Minimal complexity and boilerplate")
    print("   • Clean, maintainable architecture")
    print("=" * 50)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Path to the simple agentic Streamlit app
    streamlit_app = current_dir / "simple_agentic_streamlit.py"
    
    if not streamlit_app.exists():
        logger.error("streamlit_app_not_found", path=str(streamlit_app))
        print(f"❌ Error: {streamlit_app} not found!")
        return
    
    try:
        # Launch Streamlit app
        logger.info("launching_streamlit_app", path=str(streamlit_app))
        print(f"🚀 Starting Streamlit app: {streamlit_app}")
        print("🌐 The app will open in your default browser...")
        print("🔄 Press Ctrl+C to stop the application")
        print()
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error("streamlit_launch_error", error=str(e))
        print(f"❌ Error launching Streamlit: {e}")
    except KeyboardInterrupt:
        logger.info("streamlit_app_stopped_by_user")
        print("\n👋 Simple Agentic RAG System stopped by user")
    except Exception as e:
        logger.error("unexpected_error", error=str(e))
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    launch_simple_agentic_app()
