# Architecture

## Directory Structure

```
google_workspace_mcp/
├── auth/                          # Authentication system
│   ├── google_auth.py             # OAuth 2.0 flow, credential management, Google service builder
│   ├── oauth_config.py            # Centralized OAuth configuration (env → config object)
│   ├── scopes.py                  # All Google API scope constants + enabled_tools tracking
│   ├── service_decorator.py       # @require_google_service() decorator
│   ├── credential_store.py        # Abstract CredentialStore API + local_directory/GCS backends
│   ├── mcp_session_middleware.py  # Extracts OAuth 2.1 session info for MCP context
│   ├── oauth21_session_store.py   # OAuth 2.1 session/token storage interface
│   ├── oauth_callback_server.py   # Standalone OAuth callback HTTP server (stdio mode)
│   ├── oauth_responses.py         # HTML success/error response pages
│   ├── oauth_types.py             # Protocol type definitions
│   ├── external_oauth_provider.py # External OAuth 2.1 provider (ya29.* bearer tokens)
│   ├── auth_info_middleware.py    # Injects auth info into FastMCP context
│   ├── permissions.py             # Granular per-service permission parsing
│   └── port_resolver.py           # Port fallback logic for stdio callback server
│
├── core/                          # MCP server framework
│   ├── server.py                  # FastMCP server instance, OAuth 2.1 config, middleware
│   ├── config.py                  # Server config: transport mode, base URI, port
│   ├── tool_registry.py           # Tool enable/disable filtering, wrapping
│   ├── tool_tier_loader.py        # Loads tool names from tool_tiers.yaml
│   ├── tool_tiers.yaml            # Three-tier tool classification (core/extended/complete)
│   ├── cli.py                     # workspace-cli entry point
│   ├── context.py                 # FastMCP session context helpers
│   ├── attachment_storage.py      # Downloaded attachment file management (1-hour TTL)
│   ├── storage.py                 # Sanitized file store factory
│   ├── http_utils.py              # HTTP helper utilities
│   ├── log_formatter.py           # Enhanced log formatting with service colors
│   ├── warning_filters.py         # Startup warning suppression
│   ├── comments.py                # Shared comment operations (Docs, Sheets, Slides)
│   ├── api_enablement.py          # Google API enablement helpers
│   └── utils.py                   # Credentials directory permission checks
│
├── gmail/       # Gmail tools (14 tools) + helpers
├── gdrive/      # Drive tools (18 tools) + helpers
├── gcalendar/   # Calendar tools (7 tools) + helpers
├── gdocs/       # Docs tools (20 tools) + helpers, managers/, markdown writer
├── gsheets/     # Sheets tools (12 tools) + helpers
├── gslides/     # Slides tools (7 tools) + helpers
├── gforms/      # Forms tools (6 tools)
├── gtasks/      # Tasks tools (6 tools)
├── gcontacts/   # Contacts tools (8 tools)
├── gchat/       # Chat tools (6 tools)
├── gappsscript/ # Apps Script tools (13 tools)
├── gsearch/     # Custom Search tools (2 tools)
│
├── skills/      # Bundled Claude Code skill
│   └── managing-google-workspace/
│       ├── SKILL.md
│       ├── references/   # Per-service parameter docs
│       └── evaluations/  # Skill eval configs
│
├── tests/        # Test suite (1193 tests)
│   ├── core/     # Core module tests
│   ├── auth/     # Auth module tests
│   └── *.py      # Service-specific tests
│
├── main.py              # Primary entry point (full-featured)
├── fastmcp_server.py    # FastMCP Cloud entry point (OAuth 2.1 enforced)
├── pyproject.toml       # Build config, deps, scripts
├── Dockerfile           # Docker build
├── docker-compose.yml   # Docker Compose setup
├── uv.lock              # Pinned dependencies
├── server.json          # MCP server metadata
├── manifest.json        # MCP Registry manifest
├── smithery.yaml        # Smithery deployment config
├── glama.json           # Glama gateway config
└── fastmcp.json         # FastMCP Cloud deployment config
```

## Entry Points

### `main.py`

Primary server entry point. Supports:
- All transport modes (stdio, streamable-http, dual)
- `--single-user`, `--tools`, `--tool-tier`, `--read-only`, `--permissions`
- Env var fallbacks for all CLI flags
- Service account and OAuth 2.1 modes
- Google-style ASCII art banner on startup

### `fastmcp_server.py`

FastMCP Cloud-optimized entry point. Enforces:
- `MCP_ENABLE_OAUTH21=true`
- `WORKSPACE_MCP_STATELESS_MODE=true`
- `MCP_SINGLE_USER_MODE=false`
- Streamable HTTP transport only

### Entry Points (CLI)

Defined in `pyproject.toml`:
- `workspace-mcp` → `main:main`
- `workspace-cli` → `core.cli:main`

## Server Class Hierarchy

```
FastMCP (fastmcp library)
└── SecureFastMCP (core/server.py)
    - Adds WellKnownCacheControlMiddleware
    - Adds OriginValidationMiddleware
    - Adds MCPSessionMiddleware (session extraction)
    - Overrides list_tools(): injects USER_GOOGLE_EMAIL defaults
    - Overrides call_tool(): injects user_google_email before validation
```

## Decorator Pattern

Tools use `@require_google_service()` for automatic auth:

```python
@server.tool()
@require_google_service("gmail", "gmail_read")
async def search_gmail_messages(service, user_google_email: str, query: str):
    # service is an authenticated googleapiclient Resource, cached 30 min
    result = service.users().messages().list(userId="me", q=query).execute()
    return result
```

Multi-service variant:

```python
@require_multiple_services([
    {"service_type": "drive", "scopes": "drive_read", "param_name": "drive_service"},
    {"service_type": "docs", "scopes": "docs_read", "param_name": "docs_service"},
])
async def get_doc_content(drive_service, docs_service, ...):
    ...
```

## Authentication Flow

### Legacy OAuth 2.0 (stdio)

1. Tool call triggers `@require_google_service()`
2. Decorator checks credential store for email
3. If missing, calls `start_google_auth` → opens browser
4. User completes Google consent → callback to local HTTP server
5. Token stored in `~/.google_workspace_mcp/credentials/<email>.json`
6. Original tool call retried with credentials

### OAuth 2.1 (streamable-http)

1. Client connects to `http://host:8000/mcp`
2. Server returns 401 with OAuth metadata (RFC 9728)
3. Client registers via DCR → GoogleProvider handles Google OAuth
4. Bearer token obtained → sent on all requests
5. `MCPSessionMiddleware` extracts session → auth_info_middleware injects
6. Tool calls validated with per-request bearer token

### Service Account (Domain-Wide Delegation)

1. Server loads service account key at startup
2. `USER_GOOGLE_EMAIL` required (impersonation target)
3. No browser OAuth flow, no callback server
4. All API calls use delegated credentials
5. Optional `DWD_ALLOWED_DOMAINS` for per-request impersonation restrictions
