import os

# Optional key used to authenticate clients with this adapter.
API_KEY = os.environ.get("API_KEY", "")

# HuggingFace Authentication Settings
HUGGINGFACE = os.environ.get("HUGGINGFACE", "false").lower() == "true"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "") # Default to empty string, auth logic will verify if HF_MODE is true and this key is needed

# Directory for service account credential files
CREDENTIALS_DIR = os.environ.get("CREDENTIALS_DIR", "/app/credentials")

# JSON string for service account credentials (can be one or multiple comma-separated)
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get("GOOGLE_CREDENTIALS_JSON")

# API Key for Vertex Express Mode
raw_vertex_keys = os.environ.get("VERTEX_EXPRESS_API_KEY")
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(',') if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

# Region used by all Vertex Express model requests
VERTEX_EXPRESS_LOCATION = os.environ.get("VERTEX_EXPRESS_LOCATION", "global").strip() or "global"

# Opt-in tool schema migration; native requests otherwise remain byte-for-byte.
VERTEX_TOOL_SCHEMA_COMPAT = os.environ.get("VERTEX_TOOL_SCHEMA_COMPAT", "false").lower() == "true"

# Fake streaming settings for debugging/testing
FAKE_STREAMING_ENABLED = os.environ.get("FAKE_STREAMING", "false").lower() == "true"
FAKE_STREAMING_INTERVAL_SECONDS = float(os.environ.get("FAKE_STREAMING_INTERVAL", "1.0"))

# URL for the remote JSON file containing model lists
MODELS_CONFIG_URL = os.environ.get("MODELS_CONFIG_URL", "https://raw.githubusercontent.com/gzzhongqi/vertex2openai/refs/heads/main/vertexModels.json")

# Constant for the Vertex reasoning tag
VERTEX_REASONING_TAG = "vertex_think_tag"

# Round-robin credential selection strategy
ROUNDROBIN = os.environ.get("ROUNDROBIN", "false").lower() == "true"

# Safety score display setting
SAFETY_SCORE = os.environ.get("SAFETY_SCORE", "false").lower() == "true"
# Validation logic moved to app/auth.py

# Proxy settings
PROXY_URL = os.environ.get("PROXY_URL")
SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE")
