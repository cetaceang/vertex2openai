---
title: OpenAI to Gemini Adapter
emoji: 🔄☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860 # Default Port exposed by Dockerfile, used by Hugging Face Spaces
---

# OpenAI to Gemini Adapter

This service acts as a compatibility layer, providing an OpenAI-compatible API interface that translates requests to Google's Vertex AI Gemini models. This allows you to leverage the power of Gemini models (including Gemini 1.5 Pro and Flash) using tools and applications originally built for the OpenAI API.

The codebase is designed with modularity and maintainability in mind, located primarily within the [`app/`](app/) directory.

## Key Features

-   **OpenAI-Compatible Endpoints:** Provides standard [`/v1/chat/completions`](app/routes/chat_api.py:0) and [`/v1/models`](app/routes/models_api.py:0) endpoints.
-   **Broad Model Support:** Seamlessly translates requests for various Gemini models (e.g., `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`). Check the [`/v1/models`](app/routes/models_api.py:0) endpoint for currently available models based on your Vertex AI Project.
-   **Multiple Credential Management Methods:**
    -   **Vertex AI Express API Key:** Use a specific [`VERTEX_EXPRESS_API_KEY`](app/config.py:0) for simplified authentication with eligible models.
    -   **Google Cloud Service Accounts:**
        -   Provide the JSON key content directly via the [`GOOGLE_CREDENTIALS_JSON`](app/config.py:0) environment variable.
        -   Place multiple service account `.json` files in a designated directory ([`CREDENTIALS_DIR`](app/config.py:0)).
-   **Smart Credential Selection:**
    -   Uses the `ExpressKeyManager` for dedicated Vertex AI Express API key handling.
    -   Employs `CredentialManager` for robust service account management.
    -   Supports **round-robin rotation** ([`ROUNDROBIN=true`](app/config.py:0)) when multiple service account credentials are provided (either via [`GOOGLE_CREDENTIALS_JSON`](app/config.py:0) or [`CREDENTIALS_DIR`](app/config.py:0)), distributing requests across credentials.
-   **Streaming & Non-Streaming:** Handles both response types correctly.
-   **OpenAI Direct Mode Enhancements:** Includes tag-based extraction for reasoning/tool use information when interacting directly with certain OpenAI models (if configured).
-   **Dockerized:** Ready for deployment via Docker Compose locally or on platforms like Hugging Face Spaces.
-   **Centralized Configuration:** Environment variables managed via [`app/config.py`](app/config.py).

## Hugging Face Spaces Deployment (Recommended)

1.  **Create a Space:** On Hugging Face Spaces, create a new "Docker" SDK Space.
2.  **Upload Files:** Add all project files ([`app/`](app/) directory, [`.gitignore`](.gitignore), [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml), [`requirements.txt`](app/requirements.txt), etc.) to the repository.
3.  **Configure Secrets:** In Space settings -> Secrets, add:
    *   `API_KEY`: Your desired API key to protect this adapter service (required).
    *   *Choose one credential method:*
        *   `GOOGLE_CREDENTIALS_JSON`: The **full content** of your Google Cloud service account JSON key file(s). Separate multiple keys with commas if providing more than one within this variable.
        *   Or provide individual files if your deployment setup supports mounting volumes (less common on standard HF Spaces).
    *   `VERTEX_EXPRESS_API_KEY` (Optional): Add your Vertex AI Express API key if you plan to use Express Mode.
    *   `ROUNDROBIN` (Optional): Set to `true` to enable round-robin rotation for service account credentials.
    *   Other variables from the "Key Environment Variables" section can be set here to override defaults.
4.  **Deploy:** Hugging Face automatically builds and deploys the container, exposing port 7860.

## Local Docker Setup

### Prerequisites

-   Docker and Docker Compose
-   An external Docker network named `shared-net`. If it does not exist yet, create it with `docker network create shared-net`.
-   Google Cloud Project with Vertex AI enabled.
-   Credentials: Either a Vertex AI Express API Key or one or more Service Account key files.

### Credential Setup (Local)

Manage environment variables using a [`.env`](.env) file in the project root (ignored by git) or within your [`docker-compose.yml`](docker-compose.yml).

1.  **Method 1: Vertex Express API Key**
    *   Set the [`VERTEX_EXPRESS_API_KEY`](app/config.py:0) environment variable.
    *   Optionally set `VERTEX_EXPRESS_LOCATION` to route all Express models and variants (including `-openai` and `-openaisearch`) to one region. The default is `global`; regional values such as `us-central1` use the corresponding regional endpoint. Docker Compose passes both variables from your shell or `.env` file into the container.
2.  **Method 2: Service Account JSON Content**
    *   Set [`GOOGLE_CREDENTIALS_JSON`](app/config.py:0) to the full JSON content of your service account key(s). For multiple keys, separate the JSON objects with a comma (e.g., `{...},{...}`).
3.  **Method 3: Service Account Files in Directory**
    *   Ensure [`GOOGLE_CREDENTIALS_JSON`](app/config.py:0) is *not* set.
    *   Create a directory (e.g., `mkdir credentials`).
    *   Place your service account `.json` key files inside this directory.
    *   Mount this directory to `/app/credentials` in the container (as shown in the default [`docker-compose.yml`](docker-compose.yml)). The service will use files found in the directory specified by [`CREDENTIALS_DIR`](app/config.py:0) (defaults to `/app/credentials`).

### Environment Variables (`.env` file example)

```env
API_KEY="your_secure_api_key_here" # REQUIRED: Set a strong key for security

# --- Choose *ONE* primary credential method ---
# VERTEX_EXPRESS_API_KEY="your_vertex_express_key"          # Option 1: Express Key
# GOOGLE_CREDENTIALS_JSON='{"type": ...}{"type": ...}' # Option 2: JSON content (comma-separate multiple keys)
# CREDENTIALS_DIR="/app/credentials"                      # Option 3: Directory path (Default if GOOGLE_CREDENTIALS_JSON is unset, ensure volume mount in docker-compose)
# ---

# --- Optional Settings ---
# VERTEX_EXPRESS_LOCATION="global" # Region for all Express models (e.g. us-central1)
# ROUNDROBIN="true"              # Enable round-robin for Service Accounts (Method 2 or 3)
# FAKE_STREAMING="false"         # For debugging - simulate streaming
# FAKE_STREAMING_INTERVAL="1.0"  # Interval for fake streaming keep-alives
# GCP_PROJECT_ID="your-gcp-project-id" # Explicitly set GCP Project ID if needed
# GCP_LOCATION="us-central1"          # Explicitly set GCP Location if needed
```

### Running Locally

```bash
# Build from the local Dockerfile and start the service in detached mode
docker compose up -d --build
```
The service will typically be available at `http://localhost:8050` (check your [`docker-compose.yml`](docker-compose.yml)).

## API Usage

### Endpoints

-   `GET /v1/models`: Lists models accessible via the configured credentials/Vertex project.
-   `POST /v1/chat/completions`: The main endpoint for generating text, mimicking the OpenAI chat completions API.
-   `GET /`: Basic health check/status endpoint.

### Authentication

All requests to the adapter require an API key passed in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```
Replace `YOUR_API_KEY` with the value you set for the [`API_KEY`](app/config.py:0) environment variable.

### Example Request (`curl`)

```bash
curl -X POST http://localhost:8050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secure_api_key_here" \
  -d '{
    "model": "gemini-1.5-flash-latest",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Explain the difference between lists and tuples in Python."}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

*(Adjust URL and API Key as needed)*

## Credential Handling Priority

The application selects credentials in this order:

1.  **Vertex AI Express Mode:** If [`VERTEX_EXPRESS_API_KEY`](app/config.py:0) is set *and* the requested model is compatible with Express mode, this key is used via the [`ExpressKeyManager`](app/express_key_manager.py).
2.  **Service Account Credentials:** If Express mode isn't used/applicable:
    *   The [`CredentialManager`](app/credentials_manager.py) loads credentials first from the [`GOOGLE_CREDENTIALS_JSON`](app/config.py:0) environment variable (if set).
    *   If [`GOOGLE_CREDENTIALS_JSON`](app/config.py:0) is *not* set, it loads credentials from `.json` files within the [`CREDENTIALS_DIR`](app/config.py:0).
    *   If [`ROUNDROBIN`](app/config.py:0) is enabled (`true`), requests using Service Accounts will cycle through the loaded credentials. Otherwise, it typically uses the first valid credential found.

## Key Environment Variables

Managed in [`app/config.py`](app/config.py) and loaded from the environment:

-   `API_KEY`: **Required.** Secret key to authenticate requests *to this adapter*.
-   `VERTEX_EXPRESS_API_KEY`: Optional. Your Vertex AI Express API key for simplified authentication.
-   `VERTEX_EXPRESS_LOCATION`: Optional. Region for all Vertex Express model requests, including OpenAI Direct variants. Uses the global endpoint for `global` and a regional endpoint for other values (e.g. `us-central1`). Unset or blank values default to `global`.
-   `GOOGLE_CREDENTIALS_JSON`: Optional. String containing the JSON content of one or more service account keys (comma-separated for multiple). Takes precedence over `CREDENTIALS_DIR` for service accounts.
-   `CREDENTIALS_DIR`: Optional. Path *within the container* where service account `.json` files are located. Used only if `GOOGLE_CREDENTIALS_JSON` is not set. (Default: `/app/credentials`)
-   `ROUNDROBIN`: Optional. Set to `"true"` to enable round-robin selection among loaded Service Account credentials. (Default: `"false"`)
-   `GCP_PROJECT_ID`: Optional. Explicitly set the Google Cloud Project ID. If not set, attempts to infer from credentials.
-   `GCP_LOCATION`: Optional. Explicitly set the Google Cloud Location (region). If not set, attempts to infer or uses Vertex AI defaults.
-   `FAKE_STREAMING`: Optional. Set to `"true"` to simulate streaming output for testing. (Default: `"false"`)
-   `FAKE_STREAMING_INTERVAL`: Optional. Interval (seconds) for keep-alive messages during fake streaming. (Default: `1.0`)

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
