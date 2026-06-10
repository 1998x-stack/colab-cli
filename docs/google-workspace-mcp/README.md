# Google Workspace MCP Server

**Comprehensive MCP server** providing LLMs with natural language control over 12 Google Workspace services.

## Quick Summary

| Attribute | Value |
|-----------|-------|
| Package | `workspace-mcp` v1.21.1 |
| License | MIT |
| Author | Taylor Wilsdon |
| Python | >= 3.10 (tested on 3.11) |
| Transport | stdio (legacy) or streamable-http (recommended) |
| Auth | OAuth 2.0, OAuth 2.1 (PKCE), Service Account |
| Repo | <https://github.com/taylorwilsdon/google_workspace_mcp> |
| Docs | <https://workspacemcp.com> |

## Services (12 total)

| Service | Module | API |
|---------|--------|-----|
| Gmail | `gmail/` | Full read/write, labels, filters, drafts, attachments |
| Drive | `gdrive/` | Search, create, read, permissions, import/export |
| Calendar | `gcalendar/` | Events, free/busy, OOO, focus time |
| Docs | `gdocs/` | Create, edit, format, tables, comments, export |
| Sheets | `gsheets/` | Read/write cells, formatting, conditional rules |
| Slides | `gslides/` | Create, read, batch update, thumbnails |
| Forms | `gforms/` | Create, read, responses, publish settings |
| Tasks | `gtasks/` | Task lists, tasks, hierarchy |
| Contacts | `gcontacts/` | CRUD, groups, batch operations |
| Chat | `gchat/` | Spaces, messages, reactions |
| Apps Script | `gappsscript/` | Projects, deployments, execution |
| Custom Search | `gsearch/` | Programmable Search Engine |

## Install

```bash
cd google_workspace_mcp

# Install all deps including dev
uv sync --group dev

# Or just production deps
uv sync

# Run locally
uv run main.py

# Run tests
uv run pytest
```

## Quick Start (local)

```bash
# 1. Set credentials
export GOOGLE_OAUTH_CLIENT_ID="your-client-id"
export GOOGLE_OAUTH_CLIENT_SECRET="your-client-secret"
export OAUTHLIB_INSECURE_TRANSPORT=1

# 2. Launch (stdio, all services)
cd google_workspace_mcp && uv run main.py

# 3. Or with tool tier
uv run main.py --tool-tier core

# 4. Or specific services
uv run main.py --tools gmail drive calendar
```

## CLI

```bash
# List all tools
uv run workspace-cli list

# Call a tool
uv run workspace-cli call search_gmail_messages query="is:unread" max_results=5

# Remote endpoint
workspace-cli --url https://your.server/mcp list
```

## MCP Client Config

### Claude Code (HTTP mode, recommended)

```bash
# Start server
export MCP_ENABLE_OAUTH21=true
export GOOGLE_OAUTH_CLIENT_ID="..."
uv run main.py --transport streamable-http

# Add to Claude Code
claude mcp add --transport http workspace-mcp http://localhost:8000/mcp
```

### Claude Desktop (stdio)

```json
{
  "mcpServers": {
    "google_workspace": {
      "command": "uvx",
      "args": ["workspace-mcp", "--tool-tier", "core"],
      "env": {
        "GOOGLE_OAUTH_CLIENT_ID": "your-client-id",
        "GOOGLE_OAUTH_CLIENT_SECRET": "your-secret",
        "OAUTHLIB_INSECURE_TRANSPORT": "1"
      }
    }
  }
}
```

## Tool Tiers

| Tier | Tools | Description |
|------|-------|-------------|
| Core | ~60 | Essential: search, read, create, send |
| Extended | ~90 | Core + labels, folders, batch ops, formatting |
| Complete | 111 | Full API: comments, headers/footers, admin |

```bash
uv run main.py --tool-tier core      # Essential only
uv run main.py --tool-tier extended  # + management
uv run main.py --tool-tier complete  # Everything
```

## Auth Modes

| Mode | Flag/Env | Use Case |
|------|----------|----------|
| Legacy OAuth 2.0 | Default (stdio) | Local dev, single user |
| OAuth 2.1 (PKCE) | `MCP_ENABLE_OAUTH21=true` | Multi-user, production |
| Service Account | `GOOGLE_SERVICE_ACCOUNT_KEY_FILE` | Headless, server-to-server |
| External OAuth | `EXTERNAL_OAUTH21_PROVIDER=true` | Custom auth infrastructure |
| Read-only | `--read-only` | Restricted access |

## Key Architecture Decisions

- **30-min service caching**: `@require_google_service()` decorator caches authenticated service objects
- **Stateless mode**: Zero disk writes for containers (`WORKSPACE_MCP_STATELESS_MODE=true`)
- **Credential store backends**: local_directory (default), GCS for cloud
- **OAuth proxy storage**: memory (dev), disk (single-server), Valkey/Redis (distributed)
- **File safety**: Local file reads restricted to `WORKSPACE_ATTACHMENT_DIR` by default; `.env`, `.ssh/`, `.aws/` always blocked
