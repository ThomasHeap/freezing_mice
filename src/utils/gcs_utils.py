import os
from google.cloud import storage
from datetime import datetime, timedelta
import tempfile
import shutil

def get_signed_url(gcs_uri: str, expiration_minutes: int = 60) -> str:
    """
    Generate a signed URL for a GCS object
    
    Args:
        gcs_uri: GCS URI (gs://bucket/path)
        expiration_minutes: How long the signed URL should be valid (default: 60 minutes)
        
    Returns:
        Signed URL that can be accessed without authentication
    """
    try:
        # Parse GCS URI
        if not gcs_uri.startswith('gs://'):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        bucket_name, blob_name = gcs_uri[5:].split('/', 1)
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Generate signed URL
        expiration = datetime.utcnow() + timedelta(minutes=expiration_minutes)
        
        # Try to generate signed URL, fall back to public URL if signing fails
        try:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
        except Exception as e:
            print(f"Warning: Could not generate signed URL ({str(e)}), trying public URL...")
            # Fall back to public URL format
            signed_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        
        print(f"Generated signed URL for {gcs_uri} (expires in {expiration_minutes} minutes)")
        return signed_url
        
    except Exception as e:
        print(f"Error generating signed URL for {gcs_uri}: {str(e)}")
        return None

def download_gcs_file(gcs_uri: str, local_path: str = None) -> str:
    """
    Download a GCS file to local storage
    
    Args:
        gcs_uri: GCS URI (gs://bucket/path)
        local_path: Local path to save file (optional, will create temp file if not provided)
        
    Returns:
        Path to the downloaded file
    """
    try:
        # Parse GCS URI
        if not gcs_uri.startswith('gs://'):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        bucket_name, blob_name = gcs_uri[5:].split('/', 1)
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Create local path if not provided
        if local_path is None:
            # Create temp file with original extension
            file_ext = os.path.splitext(blob_name)[1]
            temp_fd, local_path = tempfile.mkstemp(suffix=file_ext)
            os.close(temp_fd)
        
        # Download the file
        blob.download_to_filename(local_path)
        print(f"Downloaded {gcs_uri} to {local_path}")
        
        return local_path
        
    except Exception as e:
        print(f"Error downloading {gcs_uri}: {str(e)}")
        return None

def cleanup_temp_file(file_path: str):
    """
    Clean up a temporary file
    
    Args:
        file_path: Path to the file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up {file_path}: {str(e)}") 