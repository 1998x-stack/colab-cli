# Authentication

## Google Cloud Setup

### 1. Create Project

Go to [Google Cloud Console](https://console.cloud.google.com/) → Create new project.

### 2. Create OAuth Credentials

APIs & Services → Credentials → Create Credentials → OAuth Client ID

- **Desktop Application**: For local dev, stdio mode, PKCE public clients
- **Web Application**: For hosted servers with fixed redirect URIs

### 3. Enable APIs

Enable each service API you need:

| Service | API |
|---------|-----|
| Calendar | `calendar-json.googleapis.com` |
| Drive | `drive.googleapis.com` |
| Gmail | `gmail.googleapis.com` |
| Docs | `docs.googleapis.com` |
| Sheets | `sheets.googleapis.com` |
| Slides | `slides.googleapis.com` |
| Forms | `forms.googleapis.com` |
| Tasks | `tasks.googleapis.com` |
| Chat | `chat.googleapis.com` |
| People (Contacts) | `people.googleapis.com` |
| Apps Script | `script.googleapis.com` |
| Custom Search | `customsearch.googleapis.com` |

## Environment Variables

### Required (at minimum)
```bash
export GOOGLE_OAUTH_CLIENT_ID="your-client-id"
```

### Auth Mode Selection

| Variable | Purpose |
|----------|---------|
| `GOOGLE_OAUTH_CLIENT_ID` | OAuth client ID (always required) |
| `GOOGLE_OAUTH_CLIENT_SECRET` | OAuth client secret (confidential clients only) |
| `MCP_ENABLE_OAUTH21` | `true` for OAuth 2.1 multi-user mode |
| `MCP_SINGLE_USER_MODE` | `true` for legacy single-user mode |
| `WORKSPACE_MCP_STATELESS_MODE` | `true` for container-friendly, no disk writes |
| `EXTERNAL_OAUTH21_PROVIDER` | `true` for external OAuth with bearer tokens |
| `GOOGLE_SERVICE_ACCOUNT_KEY_FILE` | Path to service account JSON key |
| `GOOGLE_SERVICE_ACCOUNT_KEY_JSON` | Inline service account JSON key |
| `USER_GOOGLE_EMAIL` | Default user email / impersonation target |
| `DWD_ALLOWED_DOMAINS` | Domain allowlist for per-request impersonation |

### Server Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKSPACE_MCP_BASE_URI` | `http://localhost` | Server base URL |
| `WORKSPACE_MCP_PORT` | `8000` | Listening port |
| `WORKSPACE_MCP_HOST` | `0.0.0.0` (OAuth 2.1) / `127.0.0.1` (legacy HTTP) | Bind host |
| `WORKSPACE_MCP_TRANSPORT` | `stdio` | Transport mode |
| `WORKSPACE_EXTERNAL_URL` | — | External URL for reverse proxy |
| `GOOGLE_OAUTH_REDIRECT_URI` | auto-constructed | Override callback URL |
| `OAUTH_CUSTOM_REDIRECT_URIS` | — | Additional redirect URIs |
| `OAUTH_ALLOWED_ORIGINS` | — | Additional CORS origins |
| `OAUTHLIB_INSECURE_TRANSPORT` | — | Set `1` for http:// callbacks (dev only) |

### Credential Storage

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKSPACE_MCP_CREDENTIALS_DIR` | `~/.google_workspace_mcp/credentials` | Credential directory |
| `GOOGLE_MCP_CREDENTIALS_DIR` | (alias for above) | Backward-compatible alias |
| `WORKSPACE_MCP_CREDENTIAL_STORE_BACKEND` | `local_directory` | `local_directory` or `gcs` |
| `WORKSPACE_MCP_GCS_BUCKET` | — | GCS bucket for credential storage |
| `WORKSPACE_MCP_GCS_PREFIX` | — | Optional object prefix in GCS |
| `WORKSPACE_MCP_GCS_REQUIRE_CMEK` | `false` | Require CMEK on GCS bucket |

### OAuth 2.1 Storage Backends

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKSPACE_MCP_OAUTH_PROXY_STORAGE_BACKEND` | `memory` | `memory`, `disk`, or `valkey` |
| `WORKSPACE_MCP_OAUTH_PROXY_DISK_DIRECTORY` | `~/.fastmcp/oauth-proxy` | Disk backend directory |
| `WORKSPACE_MCP_OAUTH_PROXY_VALKEY_HOST` | `localhost` | Valkey/Redis host |
| `WORKSPACE_MCP_OAUTH_PROXY_VALKEY_PORT` | `6379` | Valkey/Redis port |
| `FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY` | — | JWT signing key (required for PKCE public clients) |
| `WORKSPACE_MCP_ALLOWED_CLIENT_REDIRECT_URIS` | — | DCR redirect URI allowlist |

## Auth Mode Compatibility

| Mode | Single-User | OAuth 2.1 | Service Account | Stateless |
|------|:-----------:|:---------:|:---------------:|:---------:|
| `--single-user` | ✓ | ✗ | ✗ | ✗ |
| `MCP_ENABLE_OAUTH21=true` | ✗ | ✓ | ✗ | Optional |
| Service Account | ✗ | ✗ | ✓ | ✗ |
| External OAuth | ✗ | ✓ (required) | ✗ | ✓ |

## Credential Loading Priority

1. Environment variables (`export VAR=value`)
2. `.env` file in project root (note: `uvx` runs from temp dir, not your clone)
3. `client_secret.json` via `GOOGLE_CLIENT_SECRET_PATH`
4. Default `client_secret.json` in project root

## Security Notes

- **Never commit** `.env`, `client_secret.json`, or `.credentials/` to git
- **Prompt injection risk**: Emails, docs, and calendar events can contain hidden instructions. Only connect trusted data to an LLM.
- **OAuth callback**: In dev, uses `http://localhost:{port}/oauth2callback` (requires `OAUTHLIB_INSECURE_TRANSPORT=1`)
- **Production**: Use HTTPS, OAuth 2.1, and consider `WORKSPACE_MCP_ALLOWED_CLIENT_REDIRECT_URIS` to prevent phishing
- **Service account mode**: Grants domain-wide delegation — restrict scopes tightly and use `DWD_ALLOWED_DOMAINS`
