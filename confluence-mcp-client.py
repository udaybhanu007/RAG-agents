import asyncio
import os
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from typing import Optional


class ConfluenceMCPClient:
    """
    A client for interacting with Confluence through the MCP (Model Context Protocol) framework.
    """
    
    def __init__(self, 
                 mcp_server_url: str = "http://localhost:9000/mcp",
                 max_steps: int = 30,
                 azure_openai_deployment: str = "gpt-4o-mini",
                 azure_openai_api_version: str = "2024-12-01-preview"):
        """
        Initialize the Confluence MCP Client.
        
        Args:
            mcp_server_url: URL of the MCP Atlassian server
            max_steps: Maximum steps for the MCP agent
            azure_openai_deployment: Azure OpenAI deployment name
            azure_openai_api_version: Azure OpenAI API version
        """
        # Load environment variables
        load_dotenv()
        
        # Store configuration
        self.mcp_server_url = mcp_server_url
        self.max_steps = max_steps
        self.azure_openai_deployment = azure_openai_deployment
        self.azure_openai_api_version = azure_openai_api_version
        
        # Read environment variables
        self.confluence_url = os.getenv("CONFLUENCE_URL")
        self.confluence_username = os.getenv("CONFLUENCE_USERNAME")
        self.confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")
        self.jira_url = os.getenv("JIRA_URL")
        self.jira_username = os.getenv("JIRA_USERNAME")
        self.jira_api_token = os.getenv("JIRA_API_TOKEN")
        
        # Azure OpenAI configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Validate required environment variables
        self._validate_environment_variables()
        
        # Initialize MCP client and agent
        self.mcp_client = self._create_mcp_client()
        self.llm = self._create_azure_openai_llm()
        self.agent = MCPAgent(llm=self.llm, client=self.mcp_client, max_steps=self.max_steps)
    
    def _validate_environment_variables(self):
        """Validate that all required environment variables are set."""
        required_vars = {
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "CONFLUENCE_URL": self.confluence_url,
            "CONFLUENCE_USERNAME": self.confluence_username,
            "CONFLUENCE_API_TOKEN": self.confluence_api_token
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"The following environment variables must be set: {', '.join(missing_vars)}")
    
    def _create_mcp_client(self) -> MCPClient:
        """Create and configure the MCP client."""
        config = {
            "mcpServers": {
                "mcp-atlassian": {
                    "url": self.mcp_server_url
                }
            }
        }
        return MCPClient.from_dict(config)
    
    def _create_azure_openai_llm(self) -> AzureChatOpenAI:
        """Create and configure the Azure OpenAI LLM."""
        if not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required but not set")
            
        return AzureChatOpenAI(
            azure_deployment=self.azure_openai_deployment,
            api_version=self.azure_openai_api_version,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=SecretStr(self.azure_openai_api_key),
            model=self.azure_openai_deployment
        )
    
    async def query(self, query_text: str) -> str:
        """
        Execute a query using the MCP agent.
        
        Args:
            query_text: The query to execute
            
        Returns:
            The result of the query
        """
        result = await self.agent.run(query_text)
        return result


async def main():
    """Main function for standalone script execution."""
    # Create client instance
    client = ConfluenceMCPClient()
    
    # Run the query
    result = await client.query("Fetch all content of confluence page title MPC-POC")
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())