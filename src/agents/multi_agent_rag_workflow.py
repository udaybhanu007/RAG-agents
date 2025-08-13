import os
import ssl
import urllib3
import sys
import time
from typing import Dict, Any, Optional

# Disable SSL verification globally before any other imports
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'false'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Add the src directory to the path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
# Import our custom modules
from .workflow_state import WorkflowState, create_initial_state
from .agents import OrchestratorAgent, VectorRAGAgent, GraphRAGAgent # type: ignore
from .validation_synthesis import ValidatorAgent, AnswerSynthesisAgent

from core.observability import traceable, get_traceable_config
from core.azure_keyvault_manager import get_secret_from_keyvault
from core.security_middleware import SecurityMiddleware, SecurityViolationError
from .logging_config import configure_logging, get_logger

# Note: Environment loading is handled by azure_keyvault_manager based on Keyvalue_Enabled flag

# Configure centralized logging once at startup
configure_logging(
    log_level=os.getenv("LOG_LEVEL") or "INFO",
    enable_json=(os.getenv("ENABLE_JSON_LOGS") or "true").lower() == "true",
    enable_colors=(os.getenv("ENABLE_COLORED_LOGS") or "false").lower() == "true"
)

logger = get_logger("workflow_engine")


class MultiAgentRAGWorkflow:
    """
    Secure Multi-Agent RAG Workflow with comprehensive security validation
    
    This implements a secure RAG workflow with:
    - Orchestrator Agent (routing - owns ALL routing business logic)
    - Vector-RAG Agent (Qdrant search with BM25 hybrid)
    - Graph-RAG Agent (Neo4j queries)
    - Validator Agent (consistency checking)
    - Answer Synthesis Agent (final composition)
    - Security Middleware (input validation, sanitization)
    - Observability (metrics and logging)
    
    Security Features:
    - Query length limits (max 1000 characters)
    - Input sanitization for special characters
    - Malicious pattern detection
    - Query complexity analysis
    
    BM25 Integration:
    - BM25 is always initialized for hybrid search capabilities
    
    Focus: Secure implementation with comprehensive input validation.
    """
    
    def __init__(self):
        self.security_middleware = SecurityMiddleware()
        self.initialize_components()
        self.build_workflow()
    
    def initialize_components(self):
        """Initialize all LLMs, databases, and agents with optimized performance"""
        overall_start_time = time.time()
        try:
            logger.info("initialization_started", phase="components")
            
            # Use concurrent initialization for independent components
            import concurrent.futures
            from functools import partial
            
            # Initialize LLM first (needed by agents)
            start_time = time.time()
            azure_deployment = get_secret_from_keyvault("AZURE_OPENAI_DEPLOYMENT")
            azure_api_version = get_secret_from_keyvault("AZURE_OPENAI_API_VERSION")
            
            if not azure_deployment or not azure_api_version:
                raise ValueError("Azure OpenAI credentials not found. Required: AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")
            
            self.llm = AzureChatOpenAI(
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                temperature=0.0
            )
            logger.info("llm_initialized", time_taken=f"{time.time() - start_time:.2f}s")
            
            # Initialize components concurrently where possible
            concurrent_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit independent initialization tasks
                embeddings_future = executor.submit(self._initialize_embeddings_fast)
                qdrant_future = executor.submit(self._initialize_qdrant_client)
                neo4j_future = executor.submit(self._initialize_neo4j_driver)
                
                # Wait for results
                self.embeddings = embeddings_future.result()
                self.qdrant_client = qdrant_future.result()
                self.neo4j_driver = neo4j_future.result()
            
            logger.info("concurrent_initialization_completed", 
                       time_taken=f"{time.time() - concurrent_start_time:.2f}s")
            
            # Initialize BM25 retriever lazily (only when needed)
            self.bm25_retriever = None
            self._bm25_initialized = False
            logger.info("bm25_initialization_strategy", strategy="lazy_loading")
            
            # Initialize all agents
            start_time = time.time()
            self.orchestrator = OrchestratorAgent(llm=self.llm)
            
            collection_name = get_secret_from_keyvault("QDRANT_COLLECTION") or "documents"
            
            self.vector_rag = VectorRAGAgent(
                self.qdrant_client, 
                self.embeddings,
                collection_name=collection_name,
                llm=self.llm,
                bm25_retriever=None  # Will be set lazily when needed
            )
            # Set reference to workflow for lazy BM25 initialization
            self.vector_rag._workflow_ref = self
            self.graph_rag = GraphRAGAgent(self.neo4j_driver, self.llm)
            self.validator = ValidatorAgent()
            self.synthesizer = AnswerSynthesisAgent(self.llm)
            
            logger.info("agents_initialized", time_taken=f"{time.time() - start_time:.2f}s")
            
            total_time = time.time() - overall_start_time
            logger.info("workflow_components_initialized", 
                       total_time=f"{total_time:.2f}s",
                       bm25_strategy="lazy",
                       performance_optimized=True)
            
        except Exception as e:
            logger.error("component_initialization_failed", 
                        error=str(e),
                        total_time=f"{time.time() - overall_start_time:.2f}s")
            raise
    
    def _initialize_embeddings_fast(self):
        """Fast embeddings initialization with SSL bypass and caching"""
        try:
            start_time = time.time()
            logger.info("embeddings_initialization_started")
            
            # Use SentenceTransformer directly with SSL bypass
            import sentence_transformers
            import requests
            
            # Patch requests session to disable SSL verification
            original_request = requests.Session.request
            def patched_request(self, method, url, **kwargs):
                kwargs.setdefault('verify', False)
                return original_request(self, method, url, **kwargs)
            requests.Session.request = patched_request # type: ignore
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu', 
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info("embeddings_initialized", time_taken=f"{time.time() - start_time:.2f}s")
            return embeddings
            
        except Exception as e:
            logger.error("embeddings_initialization_failed", error=str(e))
            raise
    
    def _initialize_qdrant_client(self):
        """Initialize Qdrant client with connection pooling and quick health check"""
        try:
            start_time = time.time()
            logger.info("qdrant_initialization_started")
            
            qdrant_url = get_secret_from_keyvault("QDRANT_API_URL")
            qdrant_api_key = get_secret_from_keyvault("QDRANT_API_KEY")
            
            if not qdrant_url or not qdrant_api_key:
                raise ValueError("Qdrant credentials not found. Required: QDRANT_API_URL, QDRANT_API_KEY")
            
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=10  # Faster timeout for initialization
            )
            
            # Quick health check instead of full connection test
            # This validates the connection without heavy operations
            try:
                collections = client.get_collections()
                logger.info("qdrant_health_check_passed", collections_count=len(collections.collections))
            except Exception as e:
                logger.warning("qdrant_health_check_failed", error=str(e))
                # Continue anyway as it might still work for queries
            
            logger.info("qdrant_initialized", time_taken=f"{time.time() - start_time:.2f}s")
            return client
            
        except Exception as e:
            logger.error("qdrant_initialization_failed", error=str(e))
            raise
    
    def _initialize_neo4j_driver(self):
        """Initialize Neo4j driver with optimized settings and health check"""
        try:
            start_time = time.time()
            logger.info("neo4j_initialization_started")
            
            neo4j_uri = get_secret_from_keyvault("NEO4J_URI")
            neo4j_username = get_secret_from_keyvault("NEO4J_USERNAME")
            neo4j_password = get_secret_from_keyvault("NEO4J_PASSWORD")
            
            if not neo4j_uri or not neo4j_username or not neo4j_password:
                raise ValueError("Neo4j credentials not found. Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
            
            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_username, neo4j_password),
                # Add connection pool optimization for faster startup
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=10,  # Reduced from 50 for faster init
                connection_acquisition_timeout=10,  # Reduced from 30 seconds
                connection_timeout=10  # Add connection timeout
            )
            
            # Quick health check with timeout
            try:
                with driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    logger.info("neo4j_health_check_passed", test_result=test_value)
            except Exception as e:
                logger.warning("neo4j_health_check_failed", error=str(e))
                # Continue anyway as it might still work for queries
            
            logger.info("neo4j_initialized", time_taken=f"{time.time() - start_time:.2f}s")
            return driver
            
        except Exception as e:
            logger.error("neo4j_initialization_failed", error=str(e))
            raise
    
    def _get_bm25_retriever_lazy(self):
        """Lazy initialization of BM25 retriever - only when actually needed"""
        if not self._bm25_initialized:
            try:
                start_time = time.time()
                logger.info("bm25_lazy_initialization_started")
                
                self.bm25_retriever = self._initialize_bm25_retriever()
                self._bm25_initialized = True
                
                logger.info("bm25_lazy_initialized", 
                           time_taken=f"{time.time() - start_time:.2f}s",
                           success=self.bm25_retriever is not None)
                           
            except Exception as e:
                logger.error("bm25_lazy_initialization_failed", error=str(e))
                self.bm25_retriever = None
                self._bm25_initialized = True
        
        return self.bm25_retriever
    
    def _initialize_bm25_retriever(self):
        """Initialize BM25 retriever from Qdrant documents with optimized fetching."""
        try:
            collection_name = get_secret_from_keyvault("QDRANT_COLLECTION") or "documents"
            
            # Optimized document fetching - limit to reasonable size for faster init
            # Use smaller limit and pagination for better performance
            max_docs = 500  # Reduced from 1000 for faster initialization
            
            points, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=max_docs,
                with_payload=True,
                with_vectors=False  # Don't fetch vectors for BM25
            )
            
            # Convert to LangChain Documents with streaming processing
            documents = []
            for point in points:
                chunk = point.payload.get("chunk", "")  # type: ignore
                if chunk and chunk.strip():  # Only process non-empty chunks
                    documents.append(Document(
                        page_content=chunk,
                        metadata=point.payload.get("metadata", {})  # type: ignore
                    ))
            
            if not documents:
                logger.warning(f"No documents found in collection: {collection_name}")
                return None
            
            # Create BM25 retriever with optimized settings
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 10  # Keep reasonable default
            
            logger.info(f"BM25 initialized with {len(documents)} documents (max: {max_docs})")
            return bm25_retriever
            
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")
            return None
    
    def build_workflow(self):
        """
        Build the simple LangGraph workflow - Happy Path only
        
        Simple linear flow:
        1. Orchestrator decides routing (all routing logic is in OrchestratorAgent)
        2. Vector and/or Graph retrieval based on orchestrator's routing decision
        3. Validation
        4. Synthesis
        
        The workflow contains NO business logic - it only defines the flow structure.
        All routing decisions are delegated directly to the orchestrator agent.
        """
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add simple agent nodes
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("vector_rag", self.vector_rag_node)
        workflow.add_node("graph_rag", self.graph_rag_node)
        workflow.add_node("validator", self.validator_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Simple routing from orchestrator - delegate directly to orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            lambda state: self.orchestrator.get_workflow_routing(state),
            {
                "vector": "vector_rag",
                "graph": "graph_rag", 
                "both_vector_first": "vector_rag",
                "none": END  # Handle non-medical queries - go directly to END
            }
        )
        
        # For "both" routing, handle vector -> graph flow - delegate directly to orchestrator
        workflow.add_conditional_edges(
            "vector_rag",
            lambda state: self.orchestrator.get_post_vector_routing(state),
            {
                "continue_to_graph": "graph_rag",
                "continue_to_validator": "validator"
            }
        )
        
        # Simple paths to validator
        workflow.add_edge("graph_rag", "validator")
        
        # Validator always goes to synthesizer (happy path)
        workflow.add_edge("validator", "synthesizer")
        
        # End points
        workflow.add_edge("synthesizer", END)
        
        # Compile the workflow
        self.workflow = workflow.compile()
        
        logger.info("secure_workflow_compiled")
    
    
    def orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Orchestrator agent node - Simple routing decision
        Logs routing decisions
        """
        result = self.orchestrator.route_query(state)
        return result
    
    def vector_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Vector RAG agent node - Simple semantic search
        Performs search using Qdrant vector database
        """
        result = self.vector_rag.retrieve_documents(state)
        return result
    
    def graph_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Graph RAG agent node - Simple graph queries
        Performs knowledge graph queries using Neo4j
        """
        result = self.graph_rag.extract_and_query(state)
        return result
    
    def validator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Validator agent node - Simple consistency checking
        Happy path: validation should generally pass
        """
        result = self.validator.validate_results(state)
        return result
    
    def synthesizer_node(self, state: WorkflowState) -> WorkflowState:
        """
        Answer synthesis agent node - Final answer composition
        Creates comprehensive answer without citations
        """
        result = self.synthesizer.synthesize_answer(state)
        return result
    
    @traceable(**get_traceable_config("secure_workflow_execution"))
    def run(self, query: str) -> str:
        """
        Main method to run the secure workflow with a query
        
        Args:
            query: The user's question/query
            
        Returns:
            str: The final synthesized answer
            
        Raises:
            SecurityViolationError: If security validation fails
        """
        try:
            # Step 1: Security validation and sanitization
            sanitized_query = self.security_middleware.validate_and_sanitize_query(query)
            
            logger.info("secure_workflow_started", 
                       original_query_length=len(query),
                       sanitized_query_length=len(sanitized_query))
            
            # Step 2: Create initial state with sanitized query
            initial_state = create_initial_state(sanitized_query)
            
            # Step 3: Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Step 4: Extract final answer - handle both medical and non-medical cases
            final_answer = result.get("final_answer")  # Non-medical queries
            if not final_answer:
                final_answer = result.get("answer")  # Medical queries
            if not final_answer:
                final_answer = "No answer generated"
            
            logger.info("secure_workflow_completed", 
                       answer_length=len(final_answer))
            
            return final_answer
            
        except SecurityViolationError as e:
            logger.error("security_violation_in_workflow", 
                        error=str(e))
            raise
        except Exception as e:
            logger.error("secure_workflow_execution_failed", error=str(e))
            raise


def main():
    """
    Main function to demonstrate the secure workflow
    """
    try:
        # Initialize the secure workflow
        workflow = MultiAgentRAGWorkflow()
        
        # Example queries with different security scenarios
        test_queries = [
            #"What is SNOMED-CT Concepts?",  # Normal query
            "What is NIH Chest X-ray?",  # Normal query
            #"SELECT * FROM users WHERE id=1; DROP TABLE users;",  # SQL injection attempt
            #"What are the <script>alert('xss')</script> medical findings?",  # XSS attempt
            #"What is {}[]();--/**/\x00medical data?",  # Special characters test
        ]
        
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"\n--- Test Query {i} ---")
                print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
                print("=" * 50)
                
                # Run the workflow with security validation
                answer = workflow.run(query)
                
                print("✅ Query processed successfully!")
                print("Final Answer:")
                print(answer)
                
            except SecurityViolationError as e:
                print(f"❌ Security Violation: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        logger.error("main_execution_failed", error=str(e))
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
