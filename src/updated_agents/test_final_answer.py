#!/usr/bin/env python3
"""
Test Final Answer Generation in Enhanced Agentic RAG System
"""

import os
import sys

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from updated_agents.simple_agentic_app import EnhancedAgenticRAGApplication

def test_final_answer_generation():
    """Test that the workflow properly generates final answers"""
    
    print("🧪 TESTING FINAL ANSWER GENERATION")
    print("=" * 60)
    
    try:
        # Initialize the Enhanced Agentic RAG Application
        app = EnhancedAgenticRAGApplication()
        print("✅ Enhanced Agentic RAG Application initialized")
        
        # Test queries
        test_queries = [
            "What is the NIH Chest X-ray dataset?",
            "Tell me about medical imaging",
            "How many images are in the dataset?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test Query {i}: {query}")
            print("-" * 50)
            
            try:
                # Process the query
                result = app.process_query(query)
                
                # Check if final answer was generated
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                reasoning = result.get("reasoning_plan", {})
                
                print(f"✅ Query processed successfully!")
                print(f"📄 Answer length: {len(answer)} characters")
                print(f"📚 Sources count: {len(sources)}")
                print(f"🧠 Reasoning route: {reasoning.get('selected_route', 'unknown')}")
                
                if answer and answer.strip() and answer != "No answer generated":
                    print(f"🎯 Final Answer (first 200 chars): {answer[:200]}...")
                    print("✅ Final answer successfully generated!")
                else:
                    print("❌ No final answer generated or answer is empty!")
                    print(f"Raw answer: '{answer}'")
                
                # Show learning indicators
                agentic_indicators = result.get("agentic_indicators", {})
                print(f"🤖 Autonomous reasoning: {agentic_indicators.get('autonomous_reasoning', False)}")
                print(f"📈 Learning applied: {agentic_indicators.get('learning_applied', False)}")
                
            except Exception as e:
                print(f"❌ Query processing failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("🎉 FINAL ANSWER GENERATION TEST COMPLETED")
        
    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_answer_generation()
