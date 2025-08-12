import os
import ssl
import urllib3
import sys
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
from workflow_state import WorkflowState, create_initial_state
from agents import OrchestratorAgent, VectorRAGAgent, GraphRAGAgent # type: ignore
from validation_synthesis import ValidatorAgent, AnswerSynthesisAgent

# Try relative imports first, fall back to absolute imports
# try:
#     from ..core.observability import observability, traceable, get_traceable_config
#     from ..core.azure_keyvault_manager import get_secret_from_keyvault
#     from ..core.security_middleware import SecurityMiddleware, SecurityViolationError
# except ImportError:
    # Fall back to absolute imports
from core.observability import observability, traceable, get_traceable_config
from core.azure_keyvault_manager import get_secret_from_keyvault
from core.security_middleware import SecurityMiddleware, SecurityViolationError
from logging_config import configure_logging, get_logger
from sentence_transformers import SentenceTransformer

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
        """Initialize all LLMs, databases, and agents"""
        try:
            # Initialize LLM
            azure_deployment = get_secret_from_keyvault("AZURE_OPENAI_DEPLOYMENT")
            azure_api_version = get_secret_from_keyvault("AZURE_OPENAI_API_VERSION")
            
            if not azure_deployment or not azure_api_version:
                raise ValueError("Azure OpenAI credentials not found. Required: AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")
            
            self.llm = AzureChatOpenAI(
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                temperature=0.0
            )
            
            # Initialize embeddings
            # Use SentenceTransformer directly with SSL bypass
            import sentence_transformers
            import requests
            
            # Patch requests session to disable SSL verification
            original_request = requests.Session.request
            def patched_request(self, method, url, **kwargs):
                kwargs.setdefault('verify', False)
                return original_request(self, method, url, **kwargs)
            requests.Session.request = patched_request
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu', 
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Qdrant client
            qdrant_url = get_secret_from_keyvault("QDRANT_API_URL")
            qdrant_api_key = get_secret_from_keyvault("QDRANT_API_KEY")
            
            if not qdrant_url or not qdrant_api_key:
                raise ValueError("Qdrant credentials not found. Required: QDRANT_API_URL, QDRANT_API_KEY")
            
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            
            # Initialize Neo4j driver
            neo4j_uri = get_secret_from_keyvault("NEO4J_URI")
            neo4j_username = get_secret_from_keyvault("NEO4J_USERNAME")
            neo4j_password = get_secret_from_keyvault("NEO4J_PASSWORD")
            
            if not neo4j_uri or not neo4j_username or not neo4j_password:
                raise ValueError("Neo4j credentials not found. Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
            
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_username, neo4j_password)
            )
            
            # Initialize BM25 retriever (always initialize)
            self.bm25_retriever = self._initialize_bm25_retriever()
            logger.info("bm25_initialization_strategy", strategy="always", success=self.bm25_retriever is not None)
            
            # Initialize all agents
            self.orchestrator = OrchestratorAgent(llm=self.llm)  # Pass LLM for medical validation
            
            collection_name = get_secret_from_keyvault("QDRANT_COLLECTION") or "documents"
            
            self.vector_rag = VectorRAGAgent(
                self.qdrant_client, 
                self.embeddings,
                collection_name=collection_name,
                llm=self.llm,
                bm25_retriever=self.bm25_retriever
            )
            self.graph_rag = GraphRAGAgent(self.neo4j_driver, self.llm)
            self.validator = ValidatorAgent()  # No LLM needed for rule-based validation
            self.synthesizer = AnswerSynthesisAgent(self.llm)
            
            logger.info("workflow_components_initialized", bm25_enabled=self.bm25_retriever is not None)
            
        except Exception as e:
            logger.error("component_initialization_failed", error=str(e))
            raise
    
    def _initialize_bm25_retriever(self):
        """Initialize BM25 retriever from Qdrant documents."""
        try:
            collection_name = get_secret_from_keyvault("QDRANT_COLLECTION") or "documents"
            
            # Fetch documents from Qdrant
            points, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert to LangChain Documents
            documents = [
                Document(page_content=point.payload.get("chunk", ""), 
                        metadata=point.payload.get("metadata", {}))
                for point in points
                if point.payload.get("chunk", "").strip()
            ]
            
            if not documents:
                logger.warning(f"No documents found in collection: {collection_name}")
                return None
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 10
            
            logger.info(f"BM25 initialized with {len(documents)} documents")
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

