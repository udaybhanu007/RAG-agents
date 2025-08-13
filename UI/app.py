from flask import Flask, request, jsonify, render_template_string
import sys
import os
import time

# Add path to your workflow - add only the src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Insert at the beginning to ensure our modules are found first
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from agents.multi_agent_rag_workflow import MultiAgentRAGWorkflow
    print("✅ Successfully imported MultiAgentRAGWorkflow")
except ImportError as e:
    print(f"❌ Error importing MultiAgentRAGWorkflow: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    print("Make sure you're running from the correct directory and all dependencies are installed")
    sys.exit(1)

app = Flask(__name__)

# Global workflow instance
workflow = None
workflow_ready = False

def initialize_workflow():
    """Initialize the Multi-Agent RAG Workflow"""
    global workflow, workflow_ready
    
    try:
        print("🚀 Initializing Multi-Agent RAG Workflow...")
        workflow = MultiAgentRAGWorkflow()
        workflow_ready = True
        print("✅ Multi-Agent RAG Workflow ready!")
        
    except Exception as e:
        print(f"❌ Workflow initialization failed: {str(e)}")
        workflow_ready = False

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent RAG Query Search</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 50px auto; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        textarea { 
            width: 100%; 
            height: 120px; 
            padding: 15px; 
            margin: 10px 0;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            font-family: Arial, sans-serif;
        }
        textarea:focus {
            border-color: #007bff;
            outline: none;
        }
        button { 
            padding: 12px 25px; 
            margin: 5px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result { 
            margin: 20px 0; 
            padding: 20px; 
            background: #f8f9fa; 
            border-left: 4px solid #007bff;
            border-radius: 5px;
        }
        .loading { 
            color: #666; 
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            border-left-color: #dc3545;
        }
        .examples {
            margin: 20px 0;
            padding: 15px;
            background: #e9ecef;
            border-radius: 5px;
        }
        .example-item {
            margin: 5px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        .example-item:hover {
            background: #007bff;
            color: white;
        }
        .status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
        }
        .status.ready { background: #28a745; }
        .status.loading { background: #ffc107; color: #333; }
    </style>
</head>
<body>
    <div class="status" id="status">Initializing...</div>
    
    <div class="container">
        <h1>🧠 Multi-Agent RAG Query Search</h1>
        
        <textarea id="query" placeholder="Enter your query here..."></textarea><br>
        
        <button onclick="search()" id="searchBtn">🔍 Search</button>
        <button onclick="clearAll()">🗑️ Clear</button>
        
        <div class="examples">
            <strong>📝 Example Queries:</strong>
            <div class="example-item" onclick="useExample('Provide concerns about the image label accuracy in medical imaging')">
                Provide concerns about the image label accuracy in medical imaging
            </div>
            <div class="example-item" onclick="useExample('What is NIH Chest X-ray?')">
                What is NIH Chest X-ray?
            </div>
            <div class="example-item" onclick="useExample('Total number of male patients ,age is 30, Finding Labels is effusion')">
                Total number of male patients ,age is 30, Finding Labels is effusion
            </div>
            <div class="example-item" onclick="useExample('Total number of Female patients ,age less than 30, Finding Labels is effusion')">
                Total number of Female patients ,age less than 30, Finding Labels is effusion
            </div>
        </div>
        
        <div id="result"></div>
    </div>
    
    <script>
        // Check status on load
        window.addEventListener('load', checkStatus);
        
        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                updateStatus(data.ready);
            } catch (error) {
                updateStatus(false);
            }
        }
        
        function updateStatus(ready) {
            const status = document.getElementById('status');
            if (ready) {
                status.className = 'status ready';
                status.textContent = '✅ Ready';
            } else {
                status.className = 'status loading';
                status.textContent = '⏳ Loading...';
            }
        }
        
        async function search() {
            const query = document.getElementById('query').value;
            const resultDiv = document.getElementById('result');
            const searchBtn = document.getElementById('searchBtn');
            
            if (!query.trim()) {
                alert('Please enter a query');
                return;
            }
            
            searchBtn.disabled = true;
            searchBtn.textContent = '⏳ Processing...';
            resultDiv.innerHTML = '<div class="loading">🔄 Running Multi-Agent RAG Pipeline...</div>';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="result">
                            <h3>📝 Query:</h3>
                            <p style="font-style: italic; color: #666;">${escapeHtml(data.query)}</p>
                            <h3>🧠 Answer:</h3>
                            <p>${escapeHtml(data.answer)}</p>
                            <small style="color: #666;">⏱️ Processing time: ${data.time}s</small>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${escapeHtml(data.error)}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">❌ Network Error: ${escapeHtml(error.message)}</div>`;
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = '🔍 Search';
            }
        }
        
        function clearAll() {
            document.getElementById('query').value = '';
            document.getElementById('result').innerHTML = '';
        }
        
        function useExample(text) {
            document.getElementById('query').value = text;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Allow Enter key to search (Ctrl+Enter)
        document.getElementById('query').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                search();
            }
        });
        
        // Check status every 10 seconds
        setInterval(checkStatus, 10000);
    </script>
</body>
</html>
"""

# Initialize workflow at module level
print("🚀 Initializing workflow at startup...")
initialize_workflow()

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search():
    """Main search endpoint - calls your workflow.run(query)"""
    global workflow, workflow_ready
    
    # Check if workflow is ready
    if not workflow_ready:
        return jsonify({
            'success': False, 
            'error': 'System is still initializing. Please wait...'
        })
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        if len(query) > 2000:
            return jsonify({'success': False, 'error': 'Query too long (max 2000 characters)'})
        
        start = time.time()
        
        print(f"🔍 Processing query: {query[:100]}...")
        print(f"📊 Workflow ready status: {workflow_ready}")
        print(f"🧠 Workflow instance: {workflow is not None}")
        
        # Add detailed logging for debugging
        try:
            # THIS IS THE KEY LINE - Your exact workflow call
            print("🚀 Calling workflow.run()...")
            answer = workflow.run(query)
            print(f"✅ Workflow.run() completed. Answer: {answer[:100] if answer else 'None'}...")
        except Exception as workflow_error:
            print(f"❌ Workflow.run() failed: {str(workflow_error)}")
            raise workflow_error
        
        end = time.time()
        
        print(f"✅ Query processed successfully in {round(end - start, 2)}s")
        
        return jsonify({
            'success': True,
            'query': query,
            'answer': answer,
            'time': round(end - start, 2)
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'ready': workflow_ready,
        'status': 'ready' if workflow_ready else 'initializing'
    })

@app.route('/debug-query', methods=['POST'])
def debug_query():
    """Debug endpoint to analyze query processing step by step"""
    global workflow, workflow_ready
    
    if not workflow_ready:
        return jsonify({
            'error': 'System is still initializing. Please wait...'
        })
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'})
        
        print(f"\n🔍 DEBUG ANALYSIS FOR QUERY: '{query}'")
        print("=" * 80)
        
        # Test orchestrator routing directly
        from workflow_state import create_initial_state
        initial_state = create_initial_state(query)
        orchestrator_result = workflow.orchestrator.route_query(initial_state)
        
        debug_info = {
            'query': query,
            'orchestrator_route': orchestrator_result.get('route', 'unknown'),
            'orchestrator_full_result': orchestrator_result,
        }
        
        # Test Neo4j connection and data directly
        try:
            with workflow.neo4j_driver.session() as session:
                # Check if we have young male patients
                young_males_query = """
                MATCH (p:Patient) 
                WHERE p.age < 20 AND (p.gender = 'male' OR p.gender = 'Male' OR p.gender = 'M')
                RETURN p.patient_id, p.age, p.gender, p.name 
                LIMIT 10
                """
                result = session.run(young_males_query)
                young_males = [record.data() for record in result]
                debug_info['young_male_patients'] = young_males
                debug_info['young_male_count'] = len(young_males)
                
                # Check total patient count
                total_patients_query = "MATCH (p:Patient) RETURN count(p) as total"
                result = session.run(total_patients_query)
                debug_info['total_patients'] = result.single()['total']
                
                # Check medical history relationships
                medical_history_query = """
                MATCH (p:Patient)-[r]-(n) 
                WHERE p.age < 20 
                RETURN type(r) as relationship_type, labels(n) as node_labels, count(*) as count
                ORDER BY count DESC
                LIMIT 5
                """
                result = session.run(medical_history_query)
                debug_info['medical_relationships'] = [record.data() for record in result]
                
        except Exception as neo4j_error:
            debug_info['neo4j_error'] = str(neo4j_error)
        
        # Test if Graph RAG Agent would be called
        route = orchestrator_result.get('route', 'unknown')
        debug_info['will_use_graph_rag'] = route in ['graph', 'both']
        
        # Test vector search capability
        try:
            vector_state = workflow.vector_rag.retrieve_documents(initial_state)
            debug_info['vector_context_length'] = len(vector_state.get('vector_context', ''))
            debug_info['vector_search_successful'] = True
        except Exception as vector_error:
            debug_info['vector_error'] = str(vector_error)
            debug_info['vector_search_successful'] = False
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': str(e), 'debug_failed': True})

@app.route('/test-neo4j')
def test_neo4j():
    """Test Neo4j connectivity and data structure"""
    global workflow, workflow_ready
    
    if not workflow_ready:
        return jsonify({'error': 'System not ready'})
    
    try:
        with workflow.neo4j_driver.session() as session:
            # Test basic connectivity
            result = session.run("RETURN 'Neo4j Connected' as status")
            status = result.single()['status']
            
            # Get database stats
            stats_query = """
            MATCH (p:Patient) 
            RETURN 
                count(p) as total_patients,
                min(p.age) as min_age,
                max(p.age) as max_age,
                collect(DISTINCT p.gender)[0..5] as sample_genders
            """
            result = session.run(stats_query)
            stats = result.single().data()
            
            # Get sample data
            sample_query = """
            MATCH (p:Patient) 
            RETURN p.patient_id, p.age, p.gender, p.name 
            LIMIT 5
            """
            result = session.run(sample_query)
            samples = [record.data() for record in result]
            
            return jsonify({
                'status': status,
                'database_stats': stats,
                'sample_patients': samples
            })
            
    except Exception as e:
        return jsonify({'error': str(e), 'neo4j_connection_failed': True})

if __name__ == '__main__':
    print("🌟 Starting Multi-Agent RAG UI Server...")
    print("📂 Make sure you have Flask installed: pip install Flask")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
