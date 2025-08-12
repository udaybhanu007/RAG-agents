import os
import logging
from typing import Optional, Dict, List
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment files to check configuration
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_file_path = os.path.join(project_root, '.env')
env_dev_file_path = os.path.join(project_root, '.env.dev')

# Load .env first (for configuration)
if os.path.exists(env_file_path):
    load_dotenv(env_file_path)
    print(f"Loaded .env from: {env_file_path}")

# Check if Key Vault is enabled
KEYVALUE_ENABLED = os.environ.get('Keyvalue_Enabled', 'true').lower() == 'true'

# If Key Vault is disabled, load .env.dev for actual values
if not KEYVALUE_ENABLED:
    if os.path.exists(env_dev_file_path):
        load_dotenv(env_dev_file_path, override=True)
        print(f"Key Vault disabled. Loaded environment values from: {env_dev_file_path}")
        logger.info(f"Key Vault disabled. Loaded environment values from: {env_dev_file_path}")
    else:
        print(f"Key Vault disabled but .env.dev file not found: {env_dev_file_path}")
        logger.warning(f"Key Vault disabled but .env.dev file not found: {env_dev_file_path}")
else:
    print("Key Vault enabled. Will use Azure Key Vault for secrets.")
    logger.info("Key Vault enabled. Will use Azure Key Vault for secrets.")

class AzureKeyVaultManager:
    """Azure Key Vault manager using Azure CLI authentication."""
    
    def __init__(self, vault_url: Optional[str] = None):
        self.vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
        if not self.vault_url:
            raise ValueError("AZURE_KEY_VAULT_URL must be set in environment or provided as parameter")
        
        # Use Azure CLI authentication with tenant configuration
        tenant_id = os.environ.get("AZURE_TENANT_ID")
        credential = DefaultAzureCredential(
            additionally_allowed_tenants=[tenant_id, "*"]
        )
        
        self.client = SecretClient(vault_url=self.vault_url, credential=credential)
        logger.info("Azure Key Vault client initialized with Azure CLI authentication")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret from Key Vault."""
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error(f"Error getting secret '{secret_name}': {e}")
            return None
    
    def get_multiple_secrets(self, secret_names: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple secrets from Key Vault."""
        return {name: self.get_secret(name) for name in secret_names}
    
    def list_secrets(self) -> List[str]:
        """List all secret names in the Key Vault."""
        try:
            secret_properties = self.client.list_properties_of_secrets()
            return [secret.name for secret in secret_properties if secret.name]
        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret in Key Vault."""
        try:
            self.client.set_secret(secret_name, secret_value)
            return True
        except Exception as e:
            logger.error(f"Error setting secret '{secret_name}': {e}")
            return False


# Global Key Vault manager instance
_keyvault_manager: Optional[AzureKeyVaultManager] = None

def get_keyvault_manager() -> AzureKeyVaultManager:
    """Get or create a Key Vault manager instance."""
    global _keyvault_manager
    if _keyvault_manager is None:
        _keyvault_manager = AzureKeyVaultManager()
    return _keyvault_manager


def get_secret_from_keyvault(secret_name: str) -> Optional[str]:
    """Get a single secret from Azure Key Vault or environment based on Keyvalue_Enabled flag."""
    
    # Debug information
    print(f"DEBUG: get_secret_from_keyvault called with: {secret_name}")
    print(f"DEBUG: KEYVALUE_ENABLED = {KEYVALUE_ENABLED}")
    
    # If Key Vault is disabled, read from environment (already loaded from .env.dev)
    if not KEYVALUE_ENABLED:
        # Get the secret name from environment variable or use the provided name directly
        secret_value = os.environ.get(secret_name)
        print(f"DEBUG: Looking for secret: {secret_value}")
        
        if secret_value:          
            logger.info(f"Retrieved secret '{secret_value}' from environment (.env.dev)")
            return secret_value
        else:
            print(f"DEBUG: Secret '{secret_value}' not found in environment")          
            logger.warning(f"Secret '{secret_value}' not found in environment")
            return None
    
    # Key Vault is enabled - use Azure Key Vault
    try:
        manager = get_keyvault_manager()
        # Get the secret name from environment variable or use the provided name directly
        actual_secret_name = os.environ.get(secret_name, secret_name)
        secret_value = manager.get_secret(actual_secret_name)
        if secret_value:
            logger.info(f"Retrieved secret '{actual_secret_name}' from Azure Key Vault")
            return secret_value
        else:
            logger.warning(f"Secret '{actual_secret_name}' not found in Azure Key Vault")
            return None
    except Exception as e:
        logger.error(f"Error retrieving secret '{secret_name}' from Key Vault: {e}")
        return None


def list_keyvault_secrets() -> List[str]:
    """List all available secrets in the Key Vault."""
    try:
        manager = get_keyvault_manager()
        return manager.list_secrets()
    except Exception as e:
        logger.error(f"Error listing Key Vault secrets: {e}")
        return []


# def main():
#     """Test Azure Key Vault access with conditional environment fallback."""
#     print("Azure Key Vault Test with Conditional Configuration")
#     print("=" * 55)
#     print(f"\nConfiguration Status:")
#     print("-" * 40)
#     print(f"Keyvalue_Enabled: {KEYVALUE_ENABLED}")
    
#     if KEYVALUE_ENABLED:
#         print("🔑 Using Azure Key Vault for secrets")
#         print("\nTesting Azure Key Vault Connection:")
#         print("-" * 40)
        
#         try:
#             # Test connection by listing secrets
#             secrets = list_keyvault_secrets()
#             print(f"✅ Azure Key Vault connection successful")
#             print(f"Found {len(secrets)} secrets in Key Vault:")
#             for i, name in enumerate(secrets, 1):
#                 print(f"  {i}. {name}")
#         except Exception as e:
#             print(f"❌ Azure Key Vault Error: {e}")
#     else:
#         print("📄 Using .env.dev file for secrets")
    
#     # Test getting specific secrets
#     print(f"\nTesting Secret Retrieval ({'Key Vault' if KEYVALUE_ENABLED else '.env.dev'}):")
#     print("-" * 50)
    
#     test_secrets = ["QDRANT_API_URL", "AZURE_OPENAI_API_KEY", "QDRANT_API_KEY"]
    
#     for secret_name in test_secrets:
#         print(f"\nTesting '{secret_name}':")
#         secret_value = get_secret_from_keyvault(secret_name)
#         if secret_value:
#             print(f"✅ Found: {secret_value[:30]}...")
#         else:
#             print(f"❌ Not found")
    
#     # Test multiple secrets
#     print("\nTesting Multiple Secrets:")
#     print("-" * 40)
    
#     results = get_secrets_from_keyvault(test_secrets)
#     for secret_name, value in results.items():
#         status = "✅ Found" if value else "❌ Missing"
#         print(f"  {secret_name}: {status}")
    
#     print("\n" + "=" * 55)
#     print("Test completed!")
#     print(f"💡 Tip: Set Keyvalue_Enabled=true in .env to use Azure Key Vault")
#     print(f"💡 Tip: Set Keyvalue_Enabled=false in .env to use .env.dev file")


# if __name__ == "__main__":
#     main()