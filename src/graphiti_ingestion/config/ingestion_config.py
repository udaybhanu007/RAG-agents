"""
Configuration classes for Graphiti ingestion pipeline
"""
import os
import logging
from typing import Optional
from pydantic import BaseModel, Field, validator, model_validator
from dotenv import load_dotenv

# Load environment variables from .env.dev
load_dotenv(dotenv_path=".env.dev")

logger = logging.getLogger(__name__)

class AzureBlobStorageConfig(BaseModel):
    """Azure Blob Storage configuration"""
    connection_string: Optional[str] = Field(default_factory=lambda: os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    account_name: Optional[str] = Field(default_factory=lambda: os.getenv("AZURE_STORAGE_ACCOUNT_NAME"))
    account_key: Optional[str] = Field(default_factory=lambda: os.getenv("AZURE_STORAGE_ACCOUNT_KEY"))
    container_name: str = Field(default_factory=lambda: os.getenv("AZURE_BLOB_CONTAINER_NAME"))
    
    @validator('container_name')
    def validate_container_name(cls, v):
        if not v:
            raise ValueError("AZURE_BLOB_CONTAINER_NAME must be set in .env.dev")
        return v
    
    @model_validator(mode='after')
    def validate_auth_method(self):
        connection_string = self.connection_string
        account_name = self.account_name
        account_key = self.account_key
        
        if not connection_string and not (account_name and account_key):
            raise ValueError(
                "Either AZURE_STORAGE_CONNECTION_STRING or both "
                "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set"
            )
        return self
    
    def get_connection_string(self) -> str:
        """Get connection string either directly or construct from account name/key"""
        if self.connection_string:
            return self.connection_string
        elif self.account_name and self.account_key:
            return f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"
        else:
            raise ValueError("No valid Azure Storage authentication method available")
    
    class Config:
        env_file = ".env.dev"

class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration for Graphiti"""
    endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", "https://azureopenai-genaiservice.openai.azure.com/"))
    api_key: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY"))
    deployment_name: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", "genai-ind-gpt-4o-mini"))
    embedding_deployment: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"))
    api_version: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    embedding_model: str = Field(default="text-embedding-3-small")  # Default embedding model
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("AZURE_OPENAI_API_KEY must be set in .env.dev")
        return v
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        if not v:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set in .env.dev")
        return v
    
    @validator('deployment_name')
    def validate_deployment_name(cls, v):
        if not v:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT must be set in .env.dev")
        return v
    
    class Config:
        env_file = ".env.dev"

class Neo4jConfig(BaseModel):
    """Neo4j database configuration"""
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD"))
    database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    
    @validator('uri')
    def validate_uri(cls, v):
        if not v:
            raise ValueError("NEO4J_URI must be set in .env.dev")
        return v
    
    @validator('username')
    def validate_username(cls, v):
        if not v:
            raise ValueError("NEO4J_USERNAME must be set in .env.dev")
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if not v:
            raise ValueError("NEO4J_PASSWORD must be set in .env.dev")
        return v
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            logger.info("Neo4j connection test successful")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}")
            return False
    
    def setup_graphiti_requirements(self) -> bool:
        """
        Set up required indices and constraints for Graphiti in Neo4j database
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            from neo4j import GraphDatabase
            
            logger.info("Setting up Neo4j database requirements for Graphiti...")
            
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            
            with driver.session() as session:
                # List of setup queries for Graphiti
                setup_queries = [
                    # Create fulltext index for node names and summaries (required by Graphiti)
                    # Updated syntax for Neo4j Aura 5.x
                    "CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS FOR (n) ON EACH [n.name, n.summary]",
                    
                    # Create constraints for Episodic nodes
                    "CREATE CONSTRAINT episodic_uuid IF NOT EXISTS FOR (e:Episodic) REQUIRE e.uuid IS UNIQUE",
                    
                    # Create constraints for Entity nodes  
                    "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE",
                    
                    # Create constraints for Community nodes
                    "CREATE CONSTRAINT community_uuid IF NOT EXISTS FOR (c:Community) REQUIRE c.uuid IS UNIQUE",
                    
                    # Create index for timestamps
                    "CREATE INDEX episodic_created_at IF NOT EXISTS FOR (e:Episodic) ON (e.created_at)",
                    "CREATE INDEX episodic_valid_at IF NOT EXISTS FOR (e:Episodic) ON (e.valid_at)",
                    
                    # Create index for group_id
                    "CREATE INDEX episodic_group_id IF NOT EXISTS FOR (e:Episodic) ON (e.group_id)",
                    
                    # Create index for source
                    "CREATE INDEX episodic_source IF NOT EXISTS FOR (e:Episodic) ON (e.source)",
                ]
                
                # Try alternative fulltext index syntax for different Neo4j versions
                fulltext_queries = [
                    # Correct syntax for Neo4j 5.x (Aura) with multiple labels
                    "CREATE FULLTEXT INDEX node_name_and_summary FOR (n:Entity|Episodic|Community) ON EACH [n.name, n.summary]",
                    # Fallback with single label
                    "CREATE FULLTEXT INDEX node_name_and_summary FOR (n:Entity) ON EACH [n.name, n.summary]",
                    # Legacy procedure syntax
                    "CALL db.index.fulltext.createNodeIndex('node_name_and_summary', ['Entity', 'Episodic', 'Community'], ['name', 'summary'])"
                ]
                
                # First, try to create the critical fulltext index with different syntax options
                fulltext_created = False
                for i, fulltext_query in enumerate(fulltext_queries, 1):
                    try:
                        logger.debug(f"Trying fulltext index creation attempt {i}: {fulltext_query[:60]}...")
                        session.run(fulltext_query)
                        logger.info(f"✅ Fulltext index created successfully with syntax {i}")
                        fulltext_created = True
                        break
                    except Exception as e:
                        logger.debug(f"Fulltext index attempt {i} failed: {e}")
                        if "already exists" in str(e).lower():
                            logger.info("✅ Fulltext index already exists")
                            fulltext_created = True
                            break
                        continue
                
                if not fulltext_created:
                    logger.warning("⚠️ Could not create fulltext index, continuing with other setup...")
                
                successful_queries = 0
                # Execute other setup queries (skip first one as it's handled above)
                for i, query in enumerate(setup_queries[1:], 2):
                    try:
                        logger.debug(f"Executing Neo4j setup query {i}: {query[:60]}...")
                        session.run(query)
                        successful_queries += 1
                        logger.debug(f"Query {i} completed successfully")
                    except Exception as e:
                        logger.debug(f"Query {i} failed (may already exist): {e}")
                        if "already exists" in str(e).lower():
                            successful_queries += 1
                        # Continue with other queries even if one fails
                
                logger.info(f"Neo4j setup completed: {successful_queries}/{len(setup_queries)} queries successful")
                
                # Verify critical requirements
                return self._verify_graphiti_requirements(session)
            
        except ImportError:
            logger.error("Neo4j driver not installed. Please install with: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"Failed to setup Neo4j for Graphiti: {e}")
            return False
        finally:
            if 'driver' in locals():
                driver.close()
    
    def _verify_graphiti_requirements(self, session) -> bool:
        """
        Verify that required Graphiti indices and constraints exist
        
        Args:
            session: Active Neo4j session
            
        Returns:
            True if all requirements are met
        """
        try:
            # Check if critical fulltext index exists
            fulltext_check = session.run("SHOW FULLTEXT INDEXES")
            fulltext_indices = [record["name"] for record in fulltext_check]
            
            if "node_name_and_summary" in fulltext_indices:
                logger.info("✅ Required fulltext index 'node_name_and_summary' is available")
                return True
            else:
                logger.error("❌ Required fulltext index 'node_name_and_summary' is NOT available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify Graphiti requirements: {e}")
            return False
    
    class Config:
        env_file = ".env.dev"

class GraphitiIngestionConfig(BaseModel):
    """Complete configuration for Graphiti ingestion pipeline"""
    azure_blob: AzureBlobStorageConfig = Field(default_factory=AzureBlobStorageConfig)
    azure_openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    batch_size: int = Field(default=5)  # Default batch size for processing
    max_content_length: int = Field(default=15000)  # Reduced from 50000 to prevent context overflow
    chunk_size: int = Field(default=500)  # ULTRA reduced from 1000 to 500 characters for rate limiting
    chunk_overlap: int = Field(default=50)  # Reduced overlap to match smaller chunks
    
    class Config:
        env_file = ".env.dev"
    
    @classmethod
    def load_from_env(cls) -> "GraphitiIngestionConfig":
        """Load configuration from environment variables"""
        return cls()
    
    def validate_all_configs(self) -> bool:
        """Validate all configurations are properly set and setup Neo4j for Graphiti"""
        try:
            # Validate Azure Blob Storage config
            self.azure_blob.get_connection_string()  # This validates Azure Blob config
            logger.info("✅ Azure Blob Storage configuration validated")
            
            # Test Neo4j connection
            if not self.neo4j.test_connection():
                logger.error("❌ Neo4j connection failed")
                return False
            
            # Setup Neo4j requirements for Graphiti
            if not self.neo4j.setup_graphiti_requirements():
                logger.error("❌ Neo4j Graphiti setup failed")
                return False
            
            logger.info("✅ All configurations validated and Neo4j setup completed")
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False
