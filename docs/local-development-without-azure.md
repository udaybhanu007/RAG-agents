# Local Development Without Azure Access

This guide explains how to run the RAG-agents project locally when you don't have Azure Key Vault access.

## Overview

The project uses Azure Key Vault for secure secret management in production, but provides fallback mechanisms for local development without Azure access.

## Fallback Mechanism

The `AzureKeyVaultManager` automatically falls back to environment variables when Azure Key Vault is not accessible:

1. **First Priority**: Environment variables (for local development)
2. **Second Priority**: Azure Key Vault (when available)

## Setting Up Local Development

### Option 1: Direct Environment Variables

Set secrets directly in your `.env.dev` file:

```bash
# Uncomment and set these in .env.dev for local development:
QDRANT_URL=http://localhost:6333
AZURE_OPENAI_API_KEY=your-local-openai-key
AZURE_STORAGE_ACCOUNT_KEY=your-local-storage-key
NEO4J_PASSWORD=your-local-neo4j-password
```

### Option 2: Local Services

Run services locally using Docker:

```bash
# Start local Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start local Neo4j
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### Option 3: Mock/Testing Values

For testing purposes, you can use mock values:

```bash
# Mock values for testing
QDRANT_URL=http://localhost:6333
AZURE_OPENAI_API_KEY=mock-key-for-testing
AZURE_STORAGE_ACCOUNT_KEY=mock-storage-key
NEO4J_PASSWORD=test-password
```

## How the Fallback Works

When you call `get_secret_from_keyvault("qdrant-url")`, the system:

1. **Checks environment variable**: `QDRANT_URL` (secret name converted to env var format)
2. **If found**: Uses the environment variable value
3. **If not found**: Attempts Azure Key Vault
4. **If Key Vault fails**: Returns `None` and logs helpful hints

## Environment Variable Naming Convention

Secret names are converted to environment variable names:

| Secret Name | Environment Variable |
|-------------|---------------------|
| `qdrant-url` | `QDRANT_URL` |
| `azure-openai-api-key` | `AZURE_OPENAI_API_KEY` |
| `storage-account-key` | `STORAGE_ACCOUNT_KEY` |
| `neo4j-password` | `NEO4J_PASSWORD` |

**Conversion Rules**:
- Convert to uppercase
- Replace hyphens (`-`) with underscores (`_`)

## Testing the Setup

Run the Key Vault test to verify your setup:

```bash
cd src/core
python azure_keyvault_manager.py
```

Expected output:
```
Azure Key Vault Test
==============================
Using environment variable QDRANT_URL for secret 'qdrant-url'
Found 0 secrets:
✅ qdrant-url: http://localhost:6333
```

## Troubleshooting

### Common Issues

1. **"Azure Key Vault not accessible"**
   - Expected for local development
   - Set environment variables as fallback

2. **"Tip: Set environment variable QDRANT_URL"**
   - The system is suggesting which env var to set
   - Add the suggested variable to your `.env.dev`

3. **"Error getting secret"**
   - Check if the environment variable is set correctly
   - Verify the naming convention (uppercase, underscores)

### Verification Checklist

- [ ] `.env.dev` file exists and is loaded
- [ ] Environment variables follow naming convention
- [ ] Local services are running (if using localhost URLs)
- [ ] No Azure CLI login required for local development

## Production vs Development

| Environment | Secret Source | Authentication |
|-------------|---------------|----------------|
| **Production** | Azure Key Vault | Managed Identity |
| **Development** | Environment Variables | None required |
| **Testing** | Mock Values | None required |

This approach allows developers to work locally without Azure access while maintaining security in production environments.
