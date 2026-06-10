# Deployment

## Local Development

```bash
cd google_workspace_mcp
uv sync --group dev
uv run main.py                          # stdio, all services
uv run main.py --tool-tier core         # core tools only
uv run main.py --tools gmail drive      # specific services
```

## Streamable HTTP (Recommended)

```bash
export MCP_ENABLE_OAUTH21=true
export GOOGLE_OAUTH_CLIENT_ID="..."
export GOOGLE_OAUTH_CLIENT_SECRET="..."   # omit for PKCE public client
export FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY="$(openssl rand -hex 32)"  # PKCE only
export OAUTHLIB_INSECURE_TRANSPORT=1      # dev only

uv run main.py --transport streamable-http
```

Server available at `http://localhost:8000/mcp`.

## Docker

```bash
# Build
docker build -t workspace-mcp .

# Run
docker run -p 8000:8000 \
  -e MCP_ENABLE_OAUTH21=true \
  -e GOOGLE_OAUTH_CLIENT_ID="..." \
  -e GOOGLE_OAUTH_CLIENT_SECRET="..." \
  -v $(pwd)/.env:/app/.env:ro \
  workspace-mcp

# With tool tier selection
docker run -e TOOL_TIER=core -e GOOGLE_OAUTH_CLIENT_ID="..." workspace-mcp

# With specific services
docker run -e TOOLS="gmail drive calendar" -e GOOGLE_OAUTH_CLIENT_ID="..." workspace-mcp
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  gws_mcp:
    build: .
    container_name: gws_mcp
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_MCP_CREDENTIALS_DIR=/app/store_creds
    volumes:
      - ./client_secret.json:/app/client_secret.json:ro
      - store_creds:/app/store_creds:rw
    env_file:
      - .env

volumes:
  store_creds:
```

```bash
docker compose up -d
```

## Stateless Mode (Containers)

```bash
export MCP_ENABLE_OAUTH21=true
export WORKSPACE_MCP_STATELESS_MODE=true
export GOOGLE_OAUTH_CLIENT_ID="..."

uv run main.py --transport streamable-http
```

Features:
- No file system writes (tokens in memory only)
- No debug log files
- Container-ready for Docker, Kubernetes, serverless
- Each request must include a valid Bearer token

## Reverse Proxy

When behind nginx/Cloudflare/etc.:

```bash
# Option 1: Set external URL for all OAuth endpoints
export WORKSPACE_EXTERNAL_URL="https://your-domain.com"

# Option 2: Override only callback URL
export GOOGLE_OAUTH_REDIRECT_URI="https://your-domain.com/oauth2callback"

# Additional CORS origins
export OAUTH_ALLOWED_ORIGINS="https://your-domain.com"
```

Your reverse proxy must forward:
- `/mcp` — MCP endpoint
- `/oauth2callback` — OAuth callback
- `/oauth2/*` — OAuth 2.1 endpoints
- `/.well-known/*` — OAuth metadata
- `/health` — Health check
- `/attachments/*` — Downloaded file serving

## Production Checklist

- [ ] Enable OAuth 2.1 (`MCP_ENABLE_OAUTH21=true`)
- [ ] Use HTTPS (set `WORKSPACE_EXTERNAL_URL` or run behind TLS-terminating proxy)
- [ ] Set `FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY` (for PKCE public clients)
- [ ] Set `WORKSPACE_MCP_ALLOWED_CLIENT_REDIRECT_URIS` to prevent DCR phishing
- [ ] Consider stateless mode for containerized deploys
- [ ] Use `disk` or `valkey` OAuth proxy storage (not `memory`) for multi-instance
- [ ] Set `WORKSPACE_MCP_CREDENTIAL_STORE_BACKEND=gcs` for cloud credential storage
- [ ] Use service account mode only in tightly controlled environments
- [ ] Restrict scopes: use `--tool-tier`, `--read-only`, or `--permissions`
- [ ] Enable CMEK enforcement if using GCS backend (`WORKSPACE_MCP_GCS_REQUIRE_CMEK=true`)
- [ ] Health check: `GET /health` returns `{"status": "healthy"}`
