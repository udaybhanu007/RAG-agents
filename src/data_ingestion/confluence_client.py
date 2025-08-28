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

    async def download_page_content(self, page_title: str, download_dir: Optional[str] = None) -> dict:
        """
        Download content from a Confluence page based on its title and save to downloaded_content folder.
        
        Args:
            page_title: The title of the Confluence page to download
            download_dir: Optional directory path. If None, uses project root/downloaded_content
            
        Returns:
            Dictionary with download results
        """
        try:
            # Create downloaded_content directory at project root if not provided
            if download_dir is None:
                # Get the project root directory (go up from Ingestion-POC to RAG-agents)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(current_dir)
                download_dir = os.path.join(root_dir, "downloaded_content")
            
            os.makedirs(download_dir, exist_ok=True)
            
            print(f"Downloading content for page: '{page_title}' to {download_dir}")
            
            # Fetch the page content using MCP agent
            query_text = f"Fetch all content of confluence page title {page_title}"
            content = await self.query(query_text)
            
            if not content or content.strip() == "":
                return {
                    "page_title": page_title,
                    "download_dir": download_dir,
                    "file_path": None,
                    "success": False,
                    "error": "No content received from Confluence page"
                }
            
            # Create a safe filename from the page title
            import re
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', page_title)
            safe_filename = safe_filename.strip().replace(' ', '_')
            file_path = os.path.join(download_dir, f"{safe_filename}.md")
            
            # Save content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {page_title}\n\n")
                f.write(f"*Downloaded from Confluence*\n\n")
                f.write(content)
            
            print(f"Successfully downloaded page content to: {file_path}")
            
            return {
                "page_title": page_title,
                "download_dir": download_dir,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error downloading page '{page_title}': {str(e)}"
            print(error_msg)
            return {
                "page_title": page_title,
                "download_dir": download_dir if 'download_dir' in locals() else "",
                "file_path": None,
                "success": False,
                "error": error_msg
            }

    async def download_multiple_pages(self, page_titles: list, download_dir: Optional[str] = None) -> dict:
        """
        Download content from multiple Confluence pages based on their titles.
        
        Args:
            page_titles: List of page titles to download
            download_dir: Optional directory path. If None, uses project root/downloaded_content
            
        Returns:
            Dictionary with overall download results
        """
        print(f"Starting download of {len(page_titles)} pages...")
        
        results = []
        successful_downloads = 0
        failed_downloads = 0
        
        for page_title in page_titles:
            result = await self.download_page_content(page_title, download_dir)
            results.append(result)
            
            if result["success"]:
                successful_downloads += 1
            else:
                failed_downloads += 1
                print(f"Failed to download: {page_title} - {result['error']}")
        
        overall_result = {
            "total_pages": len(page_titles),
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "download_dir": results[0]["download_dir"] if results else "",
            "individual_results": results,
            "downloaded_files": [r["file_path"] for r in results if r["success"]]
        }
        
        print(f"Download completed: {successful_downloads} successful, {failed_downloads} failed")
        return overall_result


async def main():
    """Main function for standalone script execution."""
    # Create client instance
    client = ConfluenceMCPClient()
    
    # # Example 1: Download a single page
    # print("=== Downloading Single Page ===")
    # single_result = await client.download_page_content("MPC-POC")
    # print(f"Single page result: {single_result}")
    
    # Example 2: Download multiple pages
    print("\n=== Downloading Multiple Pages ===")
    page_titles = ["MPC-POC", "User Story-Requirement"]
    multiple_result = await client.download_multiple_pages(page_titles)
    print(f"Multiple pages result: {multiple_result}")
    
    # # Example 3: Original query method (for comparison)
    # print("\n=== Original Query Method ===")
    # result = await client.query("Fetch all content of confluence page title MPC-POC")
    # print(f"Query result: {result[:200]}...")  # Show first 200 chars

if __name__ == "__main__":
    asyncio.run(main())