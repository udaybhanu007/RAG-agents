"""
Simplified Box Integration for File Downloading
Focuses on core functionality: authentication and folder file downloading

Features:
- OAuth2 authentication with automatic token management
- File downloading from specific folders
- Clean error handling and logging
- Files saved to downloaded_content folder
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
import re

from boxsdk import Client
from boxsdk.auth.oauth2 import OAuth2
from boxsdk.exception import BoxAPIException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
REDIRECT_URI = "http://localhost:8080"
TOKEN_FILE = "box_oauth_tokens.json"


class BoxClient:
    """Simplified Box client for document fetching"""
    
    def __init__(self, env_file: str = ".env"):
        """Initialize Box client with environment credentials"""
        load_dotenv(env_file)
        
        self.client_id = os.getenv('Box_Client_Id')
        self.client_secret = os.getenv('Box_Client_Secret')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Box_Client_Id and Box_Client_Secret must be set in .env file")
        
        self.client: Optional[Client] = None
        self.oauth: Optional[OAuth2] = None
        
        # Initialize authentication
        self._authenticate()
    
    def _authenticate(self):
        """Handle authentication flow with better error handling"""
        try:
            if self._load_tokens() and self._validate_authentication():
                logger.info("Using existing authentication")
                return
            
            logger.info("Existing tokens invalid or missing")
            
        except Exception as e:
            logger.warning(f"Error during token validation: {e}")
        
        # Clear any invalid tokens
        self._clear_tokens()
        
        logger.info("Starting OAuth2 authentication...")
        if not self._oauth_flow():
            raise Exception("Authentication failed")
    
    def clear_tokens_and_reauthenticate(self):
        """Manually clear tokens and force re-authentication"""
        logger.info("Manually clearing tokens and re-authenticating...")
        self._clear_tokens()
        self.client = None
        self.oauth = None
        self._authenticate()
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated"""
        return self._validate_authentication()
    
    def get_token_status(self) -> Dict[str, Union[str, bool, float]]:
        """Get current token status information"""
        try:
            if not os.path.exists(TOKEN_FILE):
                return {
                    "status": "no_tokens",
                    "authenticated": False,
                    "message": "No token file found"
                }
            
            with open(TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
            
            timestamp = token_data.get('timestamp', 0)
            token_age_hours = (time.time() - timestamp) / 3600
            
            is_auth = self._validate_authentication()
            
            return {
                "status": "valid" if is_auth else "invalid",
                "authenticated": is_auth,
                "token_age_hours": round(token_age_hours, 2),
                "has_access_token": bool(token_data.get('access_token')),
                "has_refresh_token": bool(token_data.get('refresh_token')),
                "message": "Tokens are working" if is_auth else "Tokens need refresh or re-authentication"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "authenticated": False,
                "message": f"Error checking token status: {e}"
            }
    
    def _clear_tokens(self):
        """Clear stored tokens"""
        try:
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
                logger.info("Cleared expired tokens")
        except Exception as e:
            logger.error(f"Error clearing tokens: {e}")
    
    def _load_tokens(self) -> bool:
        """Load existing tokens from file"""
        try:
            if not os.path.exists(TOKEN_FILE):
                return False
            
            with open(TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
            
            self.oauth = OAuth2(
                client_id=self.client_id,
                client_secret=self.client_secret,
                access_token=token_data.get('access_token'),
                refresh_token=token_data.get('refresh_token')
            )
            
            self.client = Client(self.oauth)
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
            
            with open(TOKEN_FILE, 'w') as f:
                json.dump(token_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving tokens: {e}")
    
    def _validate_authentication(self) -> bool:
        """Validate current authentication and refresh token if needed"""
        try:
            if not self.client or not self.oauth:
                return False
            
            # Test API call
            try:
                self.client.user().get()
                logger.info("Authentication is valid")
                return True
                
            except BoxAPIException as e:
                if e.status == 401:  # Unauthorized - token expired
                    logger.info("Access token expired, attempting to refresh...")
                    return self._refresh_token()
                else:
                    logger.error(f"API error during validation: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error validating authentication: {e}")
            return False
    
    def _refresh_token(self) -> bool:
        """Refresh the access token using refresh token"""
        try:
            if not self.oauth:
                logger.error("No OAuth object available for token refresh")
                return False
            
            # Get current refresh token from saved data
            if not os.path.exists(TOKEN_FILE):
                logger.error("No token file found for refresh")
                return False
                
            with open(TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
            
            current_refresh_token = token_data.get('refresh_token')
            if not current_refresh_token:
                logger.error("No refresh token available")
                return False
            
            # Attempt to refresh the token
            access_token, refresh_token = self.oauth.refresh(current_refresh_token)
            
            # Save the new tokens (handle None refresh_token)
            new_refresh_token = refresh_token if refresh_token else current_refresh_token
            self._save_tokens(access_token, new_refresh_token)
            
            # Update the client with new tokens
            self.oauth = OAuth2(
                client_id=self.client_id,
                client_secret=self.client_secret,
                access_token=access_token,
                refresh_token=new_refresh_token
            )
            self.client = Client(self.oauth)
            
            logger.info("Token refreshed successfully")
            return True
            
        except BoxAPIException as e:
            if e.status == 400 and 'invalid_grant' in str(e):
                logger.error("Refresh token expired or invalid - need to re-authenticate")
                self._clear_tokens()
                return False
            else:
                logger.error(f"Error refreshing token: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {e}")
            return False
    
    def _oauth_flow(self) -> bool:
        """Execute OAuth2 flow with local server"""
        auth_code = None
        server_running = True
        
        class AuthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal auth_code, server_running
                
                # Handle both root path and callback path
                if self.path.startswith('/') and ('code=' in self.path or 'state=' in self.path):
                    # Parse authorization code from any path
                    parsed = urllib.parse.urlparse(self.path)
                    query_params = urllib.parse.parse_qs(parsed.query)
                    
                    if 'code' in query_params:
                        auth_code = query_params['code'][0]
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        success_page = """
                        <html>
                        <head><title>Box Authentication</title></head>
                        <body>
                            <h1>✅ Authentication Successful!</h1>
                            <p>You can close this window and return to your application.</p>
                            <script>window.close();</script>
                        </body>
                        </html>
                        """
                        self.wfile.write(success_page.encode())
                    else:
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        error_page = """
                        <html>
                        <head><title>Box Authentication Error</title></head>
                        <body>
                            <h1>❌ Authentication Failed!</h1>
                            <p>No authorization code received.</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(error_page.encode())
                    
                    server_running = False
                else:
                    # Handle other requests
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    waiting_page = """
                    <html>
                    <head><title>Box Authentication</title></head>
                    <body>
                        <h1>⏳ Waiting for Box Authentication...</h1>
                        <p>Please complete the authentication in the Box tab.</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(waiting_page.encode())
            
            def log_message(self, format, *args):
                pass  # Suppress server logs
        
        # Start local server
        server = HTTPServer(('localhost', 8080), AuthHandler)
        server_thread = threading.Thread(target=lambda: server.serve_forever())
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Create OAuth and get authorization URL
            self.oauth = OAuth2(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            auth_url, csrf_token = self.oauth.get_authorization_url(REDIRECT_URI)
            
            print(f"Opening browser for authentication: {auth_url}")
            webbrowser.open(auth_url)
            
            # Wait for authorization code
            timeout = time.time() + 300  # 5 minutes
            while server_running and time.time() < timeout:
                time.sleep(1)
            
            server.shutdown()
            
            if not auth_code:
                logger.error("No authorization code received")
                return False
            
            # Exchange code for tokens
            access_token, refresh_token = self.oauth.authenticate(auth_code)
            self._save_tokens(access_token, refresh_token)
            
            self.client = Client(self.oauth)
            logger.info("Authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"OAuth flow failed: {e}")
            return False
    
    def find_folder(self, folder_name: str) -> Optional[Dict]:
        """Find folder by name in root directory"""
        try:
            if not self.client:
                return None
            
            root_folder = self.client.folder('0')
            items = root_folder.get_items()
            
            for item in items:
                if item.type == 'folder' and item.name == folder_name:
                    return {
                        'id': item.id,
                        'name': item.name,
                        'type': item.type
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding folder '{folder_name}': {e}")
            return None
    
    def get_folder_files(self, folder_id: str) -> List[Dict]:
        """Get all files from a folder"""
        try:
            if not self.client:
                return []
            
            folder = self.client.folder(folder_id)
            items = folder.get_items()
            
            files = []
            for item in items:
                if item.type == 'file':
                    file_info = {
                        'id': item.id,
                        'name': item.name,
                        'size': getattr(item, 'size', 0),
                        'type': item.type
                    }
                    
                    # Include all files, not just validated ones for downloading
                    files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting folder files: {e}")
            return []
    
    def download_file(self, file_id: str, file_name: str) -> str:
        """Download and save the actual file to downloaded_content folder"""
        try:
            if not self.client:
                return ""
            
            # Create downloaded_content folder if it doesn't exist
            project_root = Path(__file__).parent.parent  # Go up from Ingestion-POC to RAG-agents
            download_folder = project_root / "downloaded_content"
            download_folder.mkdir(exist_ok=True)
            
            # Clean filename for safe saving
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', file_name)
            file_path = download_folder / safe_filename
            
            # Download the file
            file_obj = self.client.file(file_id)
            file_content = file_obj.content()
            
            # Write the file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"File downloaded to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading file {file_name}: {e}")
            return ""
    
    def fetch_folder_documents(self, folder_name: str) -> Dict[str, Union[str, List, int, Dict]]:
        """
        Main method: Download all files from a folder
        
        Args:
            folder_name: Name of the folder to download files from
            
        Returns:
            Dictionary with folder info and download results
        """
        try:
            if not self.client:
                return {
                    "error": "Not authenticated",
                    "folder_name": folder_name,
                    "document_count": 0,
                    "files": []
                }
            
            # Find the folder
            folder = self.find_folder(folder_name)
            if not folder:
                return {
                    "error": f"Folder '{folder_name}' not found",
                    "folder_name": folder_name,
                    "document_count": 0,
                    "files": []
                }
            
            # Get files from folder
            files = self.get_folder_files(folder['id'])
            
            # Download each file
            processed_files = []
            for file_info in files:
                try:
                    # Download the actual file only
                    downloaded_path = self.download_file(file_info['id'], file_info['name'])
                    
                    file_data = {
                        "id": file_info['id'],
                        "name": file_info['name'],
                        "size_bytes": file_info['size'],
                        "size_mb": round(file_info['size'] / (1024 * 1024), 2),
                        "status": "success"
                    }
                    
                    # Add downloaded file path
                    if downloaded_path:
                        file_data["downloaded_file"] = downloaded_path
                    
                    processed_files.append(file_data)
                    
                except Exception as e:
                    processed_files.append({
                        "id": file_info['id'],
                        "name": file_info['name'],
                        "size_bytes": file_info['size'],
                        "size_mb": round(file_info['size'] / (1024 * 1024), 2),
                        "status": "error",
                        "error": str(e)
                    })
            
            return {
                "folder_name": folder_name,
                "folder_id": folder['id'],
                "document_count": len(processed_files),
                "files": processed_files,
                "status": "success",
                "summary": {
                    "total_files": len(processed_files),
                    "successful_downloads": len([f for f in processed_files if f.get('downloaded_file')]),
                    "errors": len([f for f in processed_files if f.get('status') == 'error'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching documents from folder '{folder_name}': {e}")
            return {
                "error": str(e),
                "folder_name": folder_name,
                "document_count": 0,
                "files": []
            }


# Alias for backward compatibility
BoxOAuth2Integration = BoxClient


def main():
    """Test the Box integration with token management"""
    try:
        print("🔗 Testing Box Integration with Token Management...")
        
        # Initialize client
        box_client = BoxClient()
        
        # Show token status
        token_status = box_client.get_token_status()
        print(f"🔑 Token Status: {token_status['status']}")
        print(f"   Authenticated: {token_status['authenticated']}")
        print(f"   Message: {token_status['message']}")
        if 'token_age_hours' in token_status:
            print(f"   Token Age: {token_status['token_age_hours']} hours")
        print()
        
        # Test folder document downloading
        folder_name = "documents-ingest"
        print(f"📂 Downloading files from folder: {folder_name}")
        
        result = box_client.fetch_folder_documents(folder_name)
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            print(f"❌ Unexpected result type: {type(result)}")
            return
        
        print(f"✅ Found {result.get('document_count', 0)} files")
        
        # Show summary statistics if available
        if result.get('summary') and isinstance(result.get('summary'), dict):
            summary = result['summary']
            summary_dict = summary if isinstance(summary, dict) else {}
            print(f"📊 Summary:")
            print(f"   📁 Total files: {summary_dict.get('total_files', 0)}")
            print(f"   💾 Successfully downloaded: {summary_dict.get('successful_downloads', 0)}")
            if summary_dict.get('errors', 0) > 0:
                print(f"   ❌ Errors: {summary_dict.get('errors', 0)}")
            print()
        
        files = result.get('files', [])
        if isinstance(files, list):
            for file_info in files:
                if isinstance(file_info, dict):
                    print(f"   📄 {file_info.get('name', 'Unknown')} ({file_info.get('size_mb', 0)} MB)")
                    
                    # Show downloaded file
                    if file_info.get('downloaded_file'):
                        print(f"      💾 Downloaded to: {Path(file_info['downloaded_file']).name}")
                        print(f"      📁 Location: downloaded_content/")
                    else:
                        print(f"      ❌ Download failed")
                        
                    print()  # Add spacing between files
        else:
            print(f"❌ Files is not a list: {type(files)}")
        
        # Final token status check
        final_token_status = box_client.get_token_status()
        print(f"🔑 Final Token Status: {final_token_status['status']}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print("\n💡 Token Management Commands:")
        print("   - box_client.clear_tokens_and_reauthenticate()  # Force re-auth")
        print("   - box_client.get_token_status()                # Check token status")
        print("   - box_client.is_authenticated()                # Quick auth check")


if __name__ == "__main__":
    main()
