import os
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment files to check configuration
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_file_path = os.path.join(project_root, '.env')
env_dev_file_path = os.path.join(project_root, '.env.dev')

# Load .env first (for configuration)
if os.path.exists(env_file_path):
    load_dotenv(env_file_path)

# Check if Key Vault is enabled
KEYVALUE_ENABLED = os.environ.get('Keyvalue_Enabled', 'false').lower() == 'true'

# If Key Vault is enabled, load .env for Key Vault secret names
# If Key Vault is disabled, load .env.dev for actual values
if KEYVALUE_ENABLED:
    if os.path.exists(env_file_path):
        load_dotenv(env_file_path, override=True)
        logger.info(f"Key Vault enabled. Loaded secret names from .env")
    else:
        logger.warning(f"Key Vault enabled but .env file not found")
else:
    if os.path.exists(env_dev_file_path):
        load_dotenv(env_dev_file_path, override=True)
        logger.info(f"Key Vault disabled. Loaded environment values from .env.dev")
    else:
        logger.warning(f"Key Vault disabled but .env.dev file not found")

class AzureKeyVaultManager:
    """Azure Key Vault manager using Azure CLI authentication."""
    
    def __init__(self, vault_url: Optional[str] = None):
        # Import Azure modules only when Key Vault is being used
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            import ssl
            import certifi
        except ImportError as e:
            raise ImportError(
                "Azure Key Vault dependencies not installed. Please install them with: "
                "pip install azure-keyvault-secrets azure-identity azure-storage-blob certifi"
            ) from e
        
        self.vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
        if not self.vault_url:
            raise ValueError("AZURE_KEY_VAULT_URL must be set in environment or provided as parameter")
        
        # Configure SSL context to use system certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Use Azure CLI authentication with tenant configuration
        tenant_id = os.environ.get("AZURE_TENANT_ID")
        credential = DefaultAzureCredential(
            additionally_allowed_tenants=[tenant_id, "*"]
        )
        
        # Create client with proper SSL handling
        try:
            self.client = SecretClient(vault_url=self.vault_url, credential=credential)
            logger.info("Azure Key Vault client initialized with Azure CLI authentication")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Key Vault client: {e}")
            raise
    
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
        if not KEYVALUE_ENABLED:
            raise RuntimeError("Key Vault is disabled. Cannot create Key Vault manager.")
        _keyvault_manager = AzureKeyVaultManager()
    return _keyvault_manager


def get_secret_from_keyvault(secret_name: str) -> Optional[str]:
    """Get a single secret from Azure Key Vault or environment based on Keyvalue_Enabled flag."""
    
    # If Key Vault is disabled, read from environment (already loaded from .env.dev)
    if not KEYVALUE_ENABLED:
        # Get the secret value directly from environment (.env.dev)
        secret_value = os.environ.get(secret_name)
        
        if secret_value:          
            logger.debug(f"Retrieved secret '{secret_name}' from environment (.env.dev)")
            return secret_value
        else:
            logger.warning(f"Secret '{secret_name}' not found in environment")
            return None
    
    # Key Vault is enabled - use Azure Key Vault with secret names from .env
    try:
        manager = get_keyvault_manager()
        # Get the Key Vault secret name from environment variable (.env file)
        keyvault_secret_name = os.environ.get(secret_name, secret_name)
        
        # Use the Key Vault secret name from .env configuration
        logger.debug(f"Using Key Vault secret name '{keyvault_secret_name}' for '{secret_name}'")
        secret_value = manager.get_secret(keyvault_secret_name)
        
        if secret_value:
            logger.debug(f"Retrieved secret '{secret_name}' from Azure Key Vault")
            return secret_value
        else:
            logger.warning(f"Secret '{secret_name}' not found in Azure Key Vault")
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

