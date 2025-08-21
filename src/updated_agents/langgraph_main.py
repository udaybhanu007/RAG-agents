"""
LangGraph Agentic Workflow Main - Demo Script

This script demonstrates the LangGraph-compliant agentic workflow
while preserving all existing business logic and capabilities.
"""

import os
import sys
from typing import Dict, Any

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from updated_agents.simple_agentic_app import EnhancedAgenticRAGApplication
from core.logging_config import get_logger

logger = get_logger("langgraph_main")

def main():
    """Main demonstration of LangGraph agentic workflow"""
    logger.info("starting_langgraph_agentic_demo")
    
    # Initialize the enhanced agentic application (now using LangGraph internally)
    app = EnhancedAgenticRAGApplication()
    
    # Initialize the system (this will use LangGraph workflow)
    init_result = app.initialize_system()
    print(f"System Initialization: {init_result}")
    
    if init_result.get("status") != "success":
        print("Failed to initialize system. Exiting.")
        return
    
    # Check system status (now includes LangGraph information)
    status = app.get_system_status()
    print(f"\nSystem Status: {status}")
    
    # Test query processing
    test_query = "What are the symptoms of pneumonia?"
    
    print(f"\nProcessing query: '{test_query}'")
    result = app.process_query(test_query)
    
    print(f"\nQuery Result:")
    print(f"Answer: {result.get('final_answer', 'No answer')}")
    print(f"Confidence: {result.get('confidence_score', 0.0)}")
    print(f"Sources: {len(result.get('sources', []))}")
    
    # Show agentic indicators
    agentic_indicators = result.get('agentic_indicators', {})
    print(f"\nAgentic Capabilities Demonstrated:")
    for capability, active in agentic_indicators.items():
        print(f"  - {capability}: {active}")
    
    # Get learning insights
    insights = app.get_learning_insights()
    print(f"\nLearning Insights: {insights}")
    
    logger.info("langgraph_agentic_demo_completed")

if __name__ == "__main__":
    main()
