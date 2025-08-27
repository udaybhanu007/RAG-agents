import asyncio
import os
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

async def main():
    load_dotenv()

    # Read environment variables
    confluence_url = os.getenv("CONFLUENCE_URL")
    confluence_username = os.getenv("CONFLUENCE_USERNAME")
    confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")
    jira_url = os.getenv("JIRA_URL")
    jira_username = os.getenv("JIRA_USERNAME")
    jira_api_token = os.getenv("JIRA_API_TOKEN")

    # Create configuration dictionary
    config = {
        "mcpServers": {
            "mcp-atlassian": {
                "url": "http://localhost:9000/mcp"
            }
        }
    }

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(config)

    # Create Azure OpenAI LLM
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not azure_openai_api_key or not azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env file")
    
    llm = AzureChatOpenAI(
        azure_deployment=azure_openai_deployment,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=SecretStr(azure_openai_api_key),
        model=azure_openai_deployment
    )

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "Fetch all content of confluence page title MPC-POC",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())