from fastapi import HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
from config import API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE # Import API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE
import os
import json
import base64

# Function to validate API key (moved from config.py)
def validate_api_key(api_key_to_validate: str) -> bool:
    """
    Validate the provided API key against the configured key.
    """
    return not API_KEY or api_key_to_validate == API_KEY


def client_key_passthrough_enabled(credential_manager, express_key_manager) -> bool:
    """Use request-local client keys only when server auth and credentials are absent."""
    return (
        not API_KEY
        and not HUGGINGFACE
        and express_key_manager.get_total_keys() == 0
        and credential_manager.get_total_credentials() == 0
    )

# API Key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Dependency for API key validation
async def get_api_key(
    authorization: Optional[str] = Header(None),
    x_ip_token: Optional[str] = Header(None, alias="x-ip-token")
):
    # Check if Hugging Face auth is enabled
    if HUGGINGFACE:  # Use HUGGINGFACE from config
        if x_ip_token is None:
            raise HTTPException(
                status_code=401, # Unauthorised - because x-ip-token is missing
                detail="Missing x-ip-token header. This header is required for Hugging Face authentication."
            )

        try:
            # Decode JWT payload
            parts = x_ip_token.split('.')
            if len(parts) < 2:
                raise ValueError("Invalid JWT format: Not enough parts to extract payload.")
            payload_encoded = parts[1]
            # Add padding if necessary, as Python's base64.urlsafe_b64decode requires it
            payload_encoded += '=' * (-len(payload_encoded) % 4)
            decoded_payload_bytes = base64.urlsafe_b64decode(payload_encoded)
            payload = json.loads(decoded_payload_bytes.decode('utf-8'))
        except ValueError as ve:
            # Log server-side for debugging, but return a generic client error
            print(f"ValueError processing x-ip-token: {ve}")
            raise HTTPException(status_code=400, detail=f"Invalid JWT format in x-ip-token: {str(ve)}")
        except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as e:
            print(f"Error decoding/parsing x-ip-token payload: {e}")
            raise HTTPException(status_code=400, detail=f"Malformed x-ip-token payload: {str(e)}")
        except Exception as e: # Catch any other unexpected errors during token processing
            print(f"Unexpected error processing x-ip-token: {e}")
            raise HTTPException(status_code=500, detail="Internal error processing x-ip-token.")

        error_in_token = payload.get("error")

        if error_in_token == "InvalidAccessToken":
            raise HTTPException(
                status_code=403,
                detail="Access denied: x-ip-token indicates 'InvalidAccessToken'."
            )
        elif error_in_token is None:  # JSON 'null' is Python's None
            # If error is null, auth is successful. Now check if HUGGINGFACE_API_KEY is configured.
            print(f"HuggingFace authentication successful via x-ip-token (error field was null).")
            return HUGGINGFACE_API_KEY # Return the configured HUGGINGFACE_API_KEY
        else:
            # Any other non-null, non-"InvalidAccessToken" value in 'error' field
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: x-ip-token indicates an unhandled error: '{error_in_token}'."
            )
    else:
        # Fallback to Bearer token authentication if HUGGINGFACE env var is not "true"
        if authorization is None:
            if not API_KEY:
                return ""
            detail_message = "Missing API key. Please include 'Authorization: Bearer YOUR_API_KEY' header."
            # Optionally, provide a hint if the HUGGINGFACE env var exists but is not "true"
            if os.getenv("HUGGINGFACE") is not None: # Check for existence, not value
                 detail_message += " (Note: HUGGINGFACE mode with x-ip-token is not currently active)."
            raise HTTPException(
                status_code=401,
                detail=detail_message
            )
        
        # Check if the header starts with "Bearer "
        scheme, _, api_key = authorization.partition(" ")
        api_key = api_key.strip()
        if scheme.lower() != "bearer" or not api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key format. Use 'Authorization: Bearer YOUR_API_KEY'"
            )
        
        # Validate the API key
        if not validate_api_key(api_key): # Call local validate_api_key
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key
