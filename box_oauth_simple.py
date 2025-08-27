"""
Simple Box Integration for fetching folder content
This version focuses on OAuth2 authentication and folder content retrieval

Features:
- OAuth2 authentication with local server
- Token persistence and automatic refresh
- Error handling and security validation
- Content preview functionality for documents-ingest folder
"""

import os
import json
import logging
import webbrowser
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from typing import Optional, Dict, List, Union
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO

# Use the original boxsdk that we know works
from boxsdk import Client
from boxsdk.auth.oauth2 import OAuth2
from boxsdk.exception import BoxAPIException

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('box_oauth_simple.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REDIRECT_URI = "http://localhost:8080"
DEFAULT_SERVER_PORT = 8080
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_MAX_WORDS = 2000
TOKEN_FILE_NAME = "box_oauth_tokens.json"

class SecurityValidator:
    """Simple security validator for file operations"""
    
    ALLOWED_FILE_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc', '.csv', '.json', '.xml'}
    MAX_FILE_SIZE_MB = 100
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate if file extension is allowed"""
        ext = Path(filename).suffix.lower()
        return ext in SecurityValidator.ALLOWED_FILE_EXTENSIONS
    
    @staticmethod
    def validate_file_size(size_bytes: int) -> bool:
        """Validate if file size is within limits"""
        size_mb = size_bytes / (1024 * 1024)
        return size_mb <= SecurityValidator.MAX_FILE_SIZE_MB

class BoxAuthenticationError(Exception):
    """Custom exception for Box authentication errors"""
    pass

class BoxOAuth2Integration:
    """Simple Box integration with proper OAuth2 redirect URI handling"""
    
    def __init__(self, env_file_path: str = ".env"):
        load_dotenv(env_file_path)
        
        # Get credentials from environment
        self.client_id = os.getenv('Box_Client_Id')
        self.client_secret = os.getenv('Box_Client_Secret')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Box_Client_Id and Box_Client_Secret must be set in .env file")
        
        self.token_file = TOKEN_FILE_NAME
        self.redirect_uri = DEFAULT_REDIRECT_URI  # This MUST match your Box app configuration
        self.client: Optional[Client] = None
        self.oauth: Optional[OAuth2] = None
        self.security_validator = SecurityValidator()
        
        # Try to load existing tokens first
        if not self._load_tokens():
            # If no valid tokens, start OAuth flow
            print("🔐 No valid authentication tokens found.")
            print("🚀 Starting OAuth2 authentication...")
            print("\n⚠️  IMPORTANT: Make sure your Box app is configured with:")
            print(f"   Redirect URI: {self.redirect_uri}")
            print("   Go to: https://app.box.com/developers/console/app")
            print()
            
            if not self.start_local_server_and_authenticate():
                raise BoxAuthenticationError("Failed to authenticate with Box")
    
    def _load_tokens(self) -> bool:
        """Load existing tokens from file"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                
                self.oauth = OAuth2(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    access_token=token_data.get('access_token'),
                    refresh_token=token_data.get('refresh_token')
                )
                
                self.client = Client(self.oauth)
                logger.info("Loaded existing tokens")
                return True
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
        
        return False
    
    def _save_tokens(self, access_token: str, refresh_token: str):
        """Save tokens to file"""
        try:
            token_data = {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'timestamp': time.time()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            logger.info("Tokens saved successfully")
        except Exception as e:
            logger.error(f"Error saving tokens: {e}")
    
    def get_authorization_url(self) -> str:
        """Get OAuth2 authorization URL with proper redirect URI"""
        self.oauth = OAuth2(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        
        # Generate authorization URL with the correct redirect URI
        auth_url, csrf_token = self.oauth.get_authorization_url(self.redirect_uri)
        
        logger.info(f"Authorization URL generated with redirect URI: {self.redirect_uri}")
        return auth_url
    
    def start_local_server_and_authenticate(self) -> bool:
        """Start local server, get auth URL, and complete OAuth flow"""
        
        auth_code_container: Dict[str, Optional[str]] = {'code': None, 'error': None}
        
        class OAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    parsed_path = urllib.parse.urlparse(self.path)
                    query = urllib.parse.parse_qs(parsed_path.query)
                    
                    code = query.get('code', [None])[0]
                    error = query.get('error', [None])[0]
                    
                    if error:
                        auth_code_container['error'] = error
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(f"<h1>Authentication Error!</h1><p>Error: {error}</p>".encode())
                    elif code:
                        auth_code_container['code'] = code
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"<h1>Authentication Successful!</h1><p>You can close this window now.</p>")
                    else:
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"<h1>Authentication Failed!</h1><p>No authorization code received.</p>")
                
                except Exception as e:
                    logger.error(f"Error in OAuth handler: {e}")
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"<h1>Server Error!</h1><p>{e}</p>".encode())
            
            def log_message(self, format, *args):
                # Suppress default server logs
                pass
        
        def run_server():
            try:
                server = HTTPServer(('localhost', DEFAULT_SERVER_PORT), OAuthHandler)
                logger.info(f"OAuth server started on http://localhost:{DEFAULT_SERVER_PORT}")
                
                # Handle requests until we get the code or an error
                while not auth_code_container['code'] and not auth_code_container['error']:
                    server.handle_request()
                
                server.server_close()
                logger.info("OAuth server stopped")
            except Exception as e:
                logger.error(f"Error running OAuth server: {e}")
                auth_code_container['error'] = str(e)
        
        try:
            # Start the server in a background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Give server a moment to start
            time.sleep(1)
            
            # Get authorization URL and open browser
            auth_url = self.get_authorization_url()
            print(f"\n🔗 Opening Box authorization page...")
            print(f"URL: {auth_url}")
            
            # Open browser
            webbrowser.open(auth_url)
            
            print(f"\n⏳ Waiting for authorization (listening on {self.redirect_uri})...")
            print("Please authorize the application in your browser.")
            
            # Wait for authorization
            timeout = DEFAULT_TIMEOUT_SECONDS
            start_time = time.time()
            
            while not auth_code_container['code'] and not auth_code_container['error']:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Authorization timeout after {timeout // 60} minutes")
                time.sleep(0.5)
            
            # Check for error
            if auth_code_container['error']:
                raise BoxAuthenticationError(f"Authorization error: {auth_code_container['error']}")
            
            # Complete authentication with the code
            auth_code = auth_code_container['code']
            if not auth_code:
                raise BoxAuthenticationError("No authorization code received")
                
            logger.info(f"Received authorization code: {auth_code[:10]}...")
            
            # Exchange code for tokens
            if not self.oauth:
                raise BoxAuthenticationError("OAuth not initialized")
                
            access_token, refresh_token = self.oauth.authenticate(auth_code)
            
            # Save tokens and create client
            self._save_tokens(access_token, refresh_token)
            self.client = Client(self.oauth)
            
            logger.info("✅ Authentication completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            print(f"❌ Authentication failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        if not self.client:
            return False
        
        try:
            # Try to get user info to verify authentication
            user = self.client.user().get()
            return True
        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return False
    
    def get_user_info(self) -> Dict:
        """Get current user information"""
        if not self.is_authenticated() or not self.client:
            raise BoxAuthenticationError("Not authenticated")
        
        try:
            user = self.client.user().get()
            return {
                'id': user.id,
                'name': user.name,
                'login': user.login,
                'created_at': user.created_at,
                'modified_at': user.modified_at
            }
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            raise
    
    def find_folder_by_name(self, folder_name: str, parent_folder_id: str = "0") -> Optional[Dict]:
        """Find a folder by name"""
        if not self.is_authenticated() or not self.client:
            raise BoxAuthenticationError("Not authenticated")
        
        try:
            parent_folder = self.client.folder(parent_folder_id)
            items = parent_folder.get_items()
            
            for item in items:
                if item.type == 'folder' and item.name.lower() == folder_name.lower():
                    return {
                        'id': item.id,
                        'name': item.name,
                        'type': item.type
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error finding folder: {e}")
            raise
    
    def get_folder_contents(self, folder_id: str) -> List[Dict]:
        """Get contents of a folder"""
        if not self.is_authenticated() or not self.client:
            raise BoxAuthenticationError("Not authenticated")
        
        try:
            folder = self.client.folder(folder_id)
            items = folder.get_items()
            
            contents = []
            for item in items:
                item_dict = {
                    'id': item.id,
                    'name': item.name,
                    'type': item.type,
                    'size': getattr(item, 'size', None)
                }
                
                # Validate file if it's a file type
                if item.type == 'file':
                    if not self.security_validator.validate_file_extension(item.name):
                        logger.warning(f"Skipping file with disallowed extension: {item.name}")
                        continue
                    
                    if item_dict['size'] and not self.security_validator.validate_file_size(item_dict['size']):
                        logger.warning(f"Skipping file that's too large: {item.name}")
                        continue
                
                contents.append(item_dict)
            
            return contents
        except Exception as e:
            logger.error(f"Error getting folder contents: {e}")
            raise
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download a file from Box"""
        if not self.is_authenticated() or not self.client:
            raise BoxAuthenticationError("Not authenticated")
        
        try:
            file = self.client.file(file_id)
            file_info = file.get()
            
            # Security validation
            if not self.security_validator.validate_file_extension(file_info.name):
                raise ValueError(f"File extension not allowed: {file_info.name}")
            
            if file_info.size and not self.security_validator.validate_file_size(file_info.size):
                raise ValueError(f"File too large: {file_info.name}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as output_file:
                file.download_to(output_file)
            
            logger.info(f"Downloaded file to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def get_file_content(self, file_id: str, max_words: int = 2000) -> str:
        """Get file content as text with word limit"""
        if not self.is_authenticated() or not self.client:
            raise BoxAuthenticationError("Not authenticated")
        
        try:
            file = self.client.file(file_id)
            file_info = file.get()
            
            # Security validation
            if not self.security_validator.validate_file_extension(file_info.name):
                raise ValueError(f"File extension not allowed: {file_info.name}")
            
            # Download to memory
            buffer = BytesIO()
            file.download_to(buffer)
            buffer.seek(0)
            
            # Extract text based on file type
            filename = file_info.name.lower()
            
            if filename.endswith('.txt'):
                content = buffer.read().decode('utf-8', errors='ignore')
            elif filename.endswith('.pdf'):
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(buffer)
                    content = ''
                    for page in reader.pages:
                        content += page.extract_text() or ''
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF processing")
                    content = "PDF content extraction requires PyPDF2"
            else:
                content = "Content preview not supported for this file type"
            
            # Limit words
            words = content.split()
            if len(words) > max_words:
                content = ' '.join(words[:max_words]) + "... [truncated]"
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return f"Error reading file: {e}"
    

    
    @staticmethod
    def validate_configuration() -> bool:
        """Validate that required environment variables are set"""
        required_vars = ['Box_Client_Id', 'Box_Client_Secret']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            print("Please set these in your .env file")
            return False
        
        return True


def main():
    """Main function for fetching content from Box documents-ingest folder"""
    import sys
    
    try:
        print("🔧 Box OAuth2 Integration - Folder Content Fetcher")
        print("=" * 55)
        
        # Load environment variables first
        load_dotenv()
        
        # Validate configuration first
        if not BoxOAuth2Integration.validate_configuration():
            return
        
        # Initialize Box integration (OAuth happens automatically)
        try:
            box = BoxOAuth2Integration()
            print("✅ Authentication successful!")
        except Exception as e:
            print(f"❌ Failed to initialize Box integration: {e}")
            return
        
        # Get user info
        try:
            user_info = box.get_user_info()
            print(f"✅ Authenticated as: {user_info['name']} ({user_info['login']})")
        except Exception as e:
            print(f"⚠️  Authentication warning: {e}")
        
        # Try to find documents-ingest folder only
        print("\n📁 Looking for 'documents-ingest' folder...")
        try:
            target_folder = box.find_folder_by_name("documents-ingest")
            if target_folder:
                print(f"✅ Found folder: {target_folder['name']} (ID: {target_folder['id']})")
            else:
                print("❌ 'documents-ingest' folder not found")
                # Show available folders
                print("\n📂 Available folders in root:")
                root_contents = box.get_folder_contents("0")
                folders = [item for item in root_contents if item['type'] == 'folder']
                for folder in folders:
                    print(f"   📁 {folder['name']} (ID: {folder['id']})")
                return
        except Exception as e:
            print(f"❌ Error accessing folders: {e}")
            return
        
        # Get folder contents
        print(f"\n📄 Getting contents of '{target_folder['name']}'...")
        try:
            contents = box.get_folder_contents(target_folder['id'])
            files = [item for item in contents if item['type'] == 'file']
            if not files:
                print("📂 No files found in the folder")
                return
            print(f"📄 Found {len(files)} files:")
            for i, file in enumerate(files, 1):
                size_mb = file['size'] / (1024 * 1024) if file['size'] else 0
                print(f"   {i}. {file['name']} ({size_mb:.2f} MB) - ID: {file['id']}")
        except Exception as e:
            print(f"❌ Error getting folder contents: {e}")
            return
        
        # Read and print first 2000 words of each file (.txt and .pdf)
        for file in files:
            print(f"\n📖 Reading '{file['name']}' (ID: {file['id']})...")
            try:
                content = box.get_file_content(file['id'], max_words=2000)
                print(content)
            except Exception as e:
                print(f"❌ Error reading file: {e}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
