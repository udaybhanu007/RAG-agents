## Guidelines

1. Donot make changes in any of the existing codebase.
2. Create a separate folder "graphiti-ingestion" for ingesting the documents only. Make sure, it will not have any logic corresponding to data retrieval.
3. Always use Azure blob store as the knowledge store, where all the source documents are kept for ingestion purpose. Use the ".env.dev" file to get the configuration details for Azure blob storage.
4. Create a separate method for fetching the documents from Azure blob store. Use BlobServiceClient as part of initialization process. Donot use any kind of existing logic for downloading and processing files from Azure Blob Storage container. 
5. Create a separate class for Graphiti injestion related logic. Use init method for incorporating all kinds of initialization logic regarding gaphiti like neo4j driver, azureopenai LLM and embedder. For embedder, use OpenAIEmbedder and embedding_model as "text-embedding-3-small". Refer the ".env.dev" file to get all the configuration details.
6. Ensure that before automated entity extraction and relationship discovery of the source documents, Graphiti should perform dynamic schema evolution based on document patterns using pydantic model. Store all these pydantic schema models in a separate folder called "document-schema".
7. For your reference, the local copies of the source socuments are available within "doc-ingestion" folder. So, Graphiti should generate the schema based on the source documents.   
 