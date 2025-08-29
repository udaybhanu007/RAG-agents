import logging
from typing import Dict, Any, List, Optional
import asyncio
import os
from datetime import datetime
from openai import AsyncAzureOpenAI

from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from ..config.ingestion_config import GraphitiIngestionConfig

# Import document container from the document_schema module
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.document_schema import DocumentContainer

logger = logging.getLogger(__name__)

class GraphitiIngestionService:
    """
    Graphiti Ingestion Service for managing document ingestion through Graphiti framework
    
    This service handles all Graphiti-related initialization and ingestion logic
    including Azure OpenAI client, LLM, embedder, and cross encoder initialization.
    """
    
    def __init__(self, config: GraphitiIngestionConfig):
        """
        Initialize Graphiti service with all required components
        
        Uses Azure OpenAI client for all components (LLM, embedder, cross encoder)
        following the reference pattern for proper initialization.
        """
        self.config = config
        self._graphiti: Optional[Graphiti] = None
        self._azure_openai_client: Optional[AsyncAzureOpenAI] = None
        self._azure_llm_config: Optional[LLMConfig] = None
        self._llm_client: Optional[OpenAIClient] = None
        self._embedder: Optional[OpenAIEmbedder] = None
        self._cross_encoder: Optional[OpenAIRerankerClient] = None
        
        self._initialize_all_components()
    
    def _initialize_all_components(self) -> None:
        """Initialize all required Graphiti components"""
        try:
            # Initialize Azure OpenAI client first
            self._initialize_azure_openai_client()
            
            # Initialize Azure LLM config
            self._initialize_azure_llm_config()
            
            # Initialize LLM client with Azure OpenAI
            self._initialize_llm_client()
            
            # Initialize embedder with Azure OpenAI
            self._initialize_embedder()
            
            # Initialize cross encoder with Azure OpenAI
            self._initialize_cross_encoder()
            
            # Initialize Graphiti core with all components
            self._initialize_graphiti_core()
            
            logger.info("All Graphiti components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti components: {e}")
            raise
    
    def _initialize_azure_openai_client(self) -> None:
        """Initialize Azure OpenAI client"""
        try:
            self._azure_openai_client = AsyncAzureOpenAI(
                api_key=self.config.azure_openai.api_key,
                api_version=self.config.azure_openai.api_version,
                azure_endpoint=self.config.azure_openai.endpoint
            )
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def _initialize_azure_llm_config(self) -> None:
        """Initialize Azure LLM config"""
        try:
            self._azure_llm_config = LLMConfig(
                small_model=self.config.azure_openai.deployment_name,
                model=self.config.azure_openai.deployment_name
            )
            logger.info("Azure LLM config initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure LLM config: {e}")
            raise
    
    def _initialize_llm_client(self) -> None:
        """Initialize LLM client with Azure OpenAI following reference pattern"""
        try:
            self._llm_client = OpenAIClient(
                self._azure_llm_config,
                None,  # cache must be None for OpenAI-based clients
                self._azure_openai_client
            )
            logger.info("LLM client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def _initialize_embedder(self) -> None:
        """Initialize embedder with Azure OpenAI following reference pattern"""
        try:
            self._embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    embedding_model=self.config.azure_openai.embedding_model
                ),
                client=self._azure_openai_client
            )
            logger.info(f"Embedder initialized with model: {self.config.azure_openai.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise
    
    def _initialize_cross_encoder(self) -> None:
        """Initialize cross encoder with Azure OpenAI following reference pattern"""
        try:
            self._cross_encoder = OpenAIRerankerClient(
                self._azure_llm_config,
                self._azure_openai_client
            )
            logger.info("Cross encoder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cross encoder: {e}")
            logger.info("Continuing without cross encoder - basic functionality will work")
            self._cross_encoder = None
    
    def _initialize_graphiti_core(self) -> None:
        """Initialize Graphiti core with all components following reference pattern"""
        try:
            self._graphiti = Graphiti(
                self.config.neo4j.uri,
                self.config.neo4j.username,
                self.config.neo4j.password,
                llm_client=self._llm_client,
                embedder=self._embedder,
                cross_encoder=self._cross_encoder
            )
            logger.info("Graphiti core initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti core: {e}")
            raise
    
    async def ingest_single_document(self, document: DocumentContainer) -> Dict[str, Any]:
        """
        Ingest a single document into Graphiti with environment-based rate limiting
        
        Args:
            document: DocumentContainer with document content and metadata
            
        Returns:
            Dict containing ingestion result information
        """
        try:
            if not self._graphiti:
                raise ValueError("Graphiti not initialized. Call initialize() first.")
            
            # Convert document to Graphiti episode format
            episode_data = document.to_graphiti_episode()
            
            # Add episode to Graphiti (rate limiting handled by environment variables)
            episode_id = await self._graphiti.add_episode(**episode_data)
            
            logger.info(f"Document ingested successfully with episode ID: {episode_id}")
            
            return {
                "success": True,
                "episode_id": episode_id,
                "document_name": document.title,
                "content_length": len(document.content),
                "ingestion_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest document {document.title}: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_name": document.title,
                "ingestion_timestamp": datetime.now().isoformat()
            }
    
    def get_graphiti_instance(self) -> Optional[Graphiti]:
        """Get the initialized Graphiti instance"""
        return self._graphiti
    
    def is_initialized(self) -> bool:
        """Check if all components are properly initialized"""
        return all([
            self._graphiti,
            self._azure_openai_client,
            self._azure_llm_config,
            self._llm_client,
            self._embedder
            # Note: cross_encoder is optional, so not included in this check
        ])
    
    async def close_connections(self) -> None:
        """Clean up resources and close connections"""
        try:
            if self._azure_openai_client:
                await self._azure_openai_client.close()
            logger.info("Graphiti ingestion service connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing Graphiti service connections: {e}")
    
    async def close(self) -> None:
        """Clean up resources (alias for close_connections)"""
        await self.close_connections()
