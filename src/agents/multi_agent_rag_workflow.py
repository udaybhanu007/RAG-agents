import os
from typing import cast, Dict, Any
from dotenv import load_dotenv
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
from observability import observability
from logging_config import configure_logging, get_logger
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure centralized logging once at startup
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_json=os.getenv("ENABLE_JSON_LOGS", "true").lower() == "true",
    enable_colors=os.getenv("ENABLE_COLORED_LOGS", "false").lower() == "true"
)

logger = get_logger("workflow_engine")


class MultiAgentRAGWorkflow:
    """
    Simple Multi-Agent RAG Workflow - Happy Path Implementation
    
    This implements the clean happy path flow:
    - Orchestrator Agent (routing - owns ALL routing business logic)
    - Vector-RAG Agent (Qdrant search with optional BM25 hybrid)
    - Graph-RAG Agent (Neo4j queries)
    - Validator Agent (consistency checking)
    - Answer Synthesis Agent (final composition)
    - Observability (metrics and logging)
    
    BM25 Integration Options:
    - eager_bm25_init=True: Initialize BM25 at startup (default)
    - eager_bm25_init=False: Disable BM25, use vector-only search
    
    Focus: Clean, simple implementation without retry complexity.
    The workflow contains NO business logic - it's a pure orchestration layer.
    All routing decisions are handled by the OrchestratorAgent.
    """
    
    def __init__(self, eager_bm25_init: bool = True):
        self.eager_bm25_init = eager_bm25_init
        self.initialize_components()
        self.build_workflow()
    
    def initialize_components(self):
        """Initialize all LLMs, databases, and agents"""
        try:
            # Initialize LLM
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                temperature=0.0
            )
            
            # Initialize embeddings
            # model_name ='all-MiniLM-L6-v2'
            # self.embeddings =SentenceTransformer(model_name)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        
            # self.embeddings = AzureOpenAIEmbeddings(
            #     azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            # )
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_API_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            
            # Initialize Neo4j driver
            self.neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(
                    os.getenv("NEO4J_USERNAME", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", "password")
                )
            )
            
            # Initialize BM25 retriever (configurable initialization)
            if self.eager_bm25_init:
                # Initialize BM25 at startup
                self.bm25_retriever = self._initialize_bm25_retriever()
                logger.info("bm25_initialization_strategy", strategy="eager", success=self.bm25_retriever is not None)
            else:
                # Skip BM25 initialization - will use vector-only search
                self.bm25_retriever = None
                logger.info("bm25_initialization_strategy", strategy="disabled")
            
            # Initialize all agents
            self.orchestrator = OrchestratorAgent(llm=self.llm)  # Pass LLM for medical validation
            self.vector_rag = VectorRAGAgent(
                self.qdrant_client, 
                self.embeddings,
                collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
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
        """
        Initialize BM25 retriever from Qdrant documents.
        Simple implementation that fetches documents and creates BM25 index.
        """
        try:
            # Basic validation
            if not hasattr(self, 'qdrant_client') or self.qdrant_client is None:
                logger.error("bm25_initialization_failed", error="qdrant_client not available")
                return None
            
            collection_name = os.getenv("QDRANT_COLLECTION", "documents")
            
            # Fetch documents from Qdrant
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=5000,  # Reasonable limit for BM25 indexing
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = scroll_result
            
            if not points:
                logger.warning("no_documents_found_for_bm25", collection=collection_name)
                return None
            
            # Convert to LangChain Documents
            documents = []
            for point in points:
                content = point.payload.get("chunk", "") # type: ignore
                metadata = point.payload.get("metadata", {}) # type: ignore
                
                if content.strip():  # Only non-empty content
                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
            
            if not documents:
                logger.warning("no_valid_documents_for_bm25")
                return None
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 10
            
            logger.info("bm25_retriever_initialized", 
                       document_count=len(documents), 
                       collection=collection_name)
            
            return bm25_retriever
            
        except Exception as e:
            logger.error("bm25_initialization_failed", error=str(e))
            return None  # Graceful fallback to vector-only search
    
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
        
        logger.info("simple_workflow_compiled_happy_path_only")
    
    
    def orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Orchestrator agent node - Simple routing decision
        Measures performance and logs routing decisions
        """
        with observability.measure_agent_performance("orch", cast(Dict[str, Any], state)):
            result = self.orchestrator.route_query(state)
            return result
    
    def vector_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Vector RAG agent node - Simple semantic search
        Performs search using Qdrant vector database
        """
        with observability.measure_agent_performance("vec", cast(Dict[str, Any], state)):
            result = self.vector_rag.retrieve_documents(state)
            return result
    
    def graph_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Graph RAG agent node - Simple graph queries
        Performs knowledge graph queries using Neo4j
        """
        with observability.measure_agent_performance("graph", cast(Dict[str, Any], state)):
            result = self.graph_rag.extract_and_query(state)
            return result
    
    def validator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Validator agent node - Simple consistency checking
        Happy path: validation should generally pass
        """
        with observability.measure_agent_performance("val", cast(Dict[str, Any], state)):
            result = self.validator.validate_results(state)
            return result
    
    def synthesizer_node(self, state: WorkflowState) -> WorkflowState:
        """
        Answer synthesis agent node - Final answer composition
        Creates comprehensive answer without citations
        """
        with observability.measure_agent_performance("ans", cast(Dict[str, Any], state)):
            result = self.synthesizer.synthesize_answer(state)
            return result
    
    def run(self, query: str) -> str:
        """
        Main method to run the workflow with a query
        
        Args:
            query: The user's question/query
            
        Returns:
            str: The final synthesized answer
        """
        try:
            # Create initial state
            initial_state = create_initial_state(query)
            
            logger.info("workflow_started", query=query)
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract final answer - handle both medical and non-medical cases
            final_answer = result.get("final_answer")  # Non-medical queries
            if not final_answer:
                final_answer = result.get("answer")  # Medical queries
            if not final_answer:
                final_answer = "No answer generated"
            
            logger.info("workflow_completed", answer_length=len(final_answer))
            
            return final_answer
            
        except Exception as e:
            logger.error("workflow_execution_failed", error=str(e))
            raise


def main():
    """
    Main function to demonstrate the workflow
    """
    try:
        # Initialize the workflow
        workflow = MultiAgentRAGWorkflow()
        
        # Example query
        #query ="what is .net?"
        #query = "What is NIH Chest X-ray?"
        query ="Provide concerns about the image label accuracy in medical"
        #query ="Tell me about the medical history of patient ID 1, including all findings and their progression"
        #query ="Tell me about the medical history of patient ID 1, including all findings and their progression?"
        print(f"Query: {query}")
        print("=" * 50)
        
        # Run the workflow
        answer = workflow.run(query)
        
        print("Final Answer:")
        print(answer)
        
    except Exception as e:
        logger.error("main_execution_failed", error=str(e))
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

