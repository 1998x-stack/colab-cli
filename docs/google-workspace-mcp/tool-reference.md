# Tool Reference

114 tools across 12 services, organized in 3 tiers.

## Tier System

| Tier | Count | Description |
|------|-------|-------------|
| **Core** | ~60 | Everyday operations: search, read, create, send |
| **Extended** | ~90 | Management: labels, folders, batch ops, formatting |
| **Complete** | 111 | Full API: comments, headers, admin, debugging |

## Gmail (14 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `search_gmail_messages` | Core | Search with Gmail operators, returns message/thread IDs |
| `get_gmail_message_content` | Core | Full message: subject, sender, body, attachments |
| `get_gmail_messages_content_batch` | Core | Batch retrieve up to 25 messages |
| `send_gmail_message` | Core | Send with HTML, CC/BCC, threading, attachments |
| `get_gmail_attachment_content` | Extended | Download attachments to local disk |
| `get_gmail_thread_content` | Extended | Complete conversation thread |
| `modify_gmail_message_labels` | Extended | Add/remove labels (archive, trash, etc.) |
| `list_gmail_labels` | Extended | All system and user labels |
| `manage_gmail_label` | Extended | Create, update, delete labels |
| `draft_gmail_message` | Extended | Create drafts with threading |
| `list_gmail_filters` | Extended | List Gmail filters |
| `manage_gmail_filter` | Extended | Create or delete filters |
| `get_gmail_threads_content_batch` | Complete | Batch retrieve threads |
| `batch_modify_gmail_message_labels` | Complete | Bulk label operations |

**OAuth scopes**: `gmail.readonly`, `gmail.send`, `gmail.compose`, `gmail.modify`, `gmail.labels`, `gmail.settings.basic`

## Google Drive (16 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `search_drive_files` | Core | Search with Drive query syntax or free text |
| `get_drive_file_content` | Core | Read Docs, Sheets, Office files |
| `get_drive_file_download_url` | Core | Download files to local disk |
| `create_drive_file` | Core | Create from content or URL |
| `create_drive_folder` | Core | Create empty folders |
| `import_to_google_doc` | Core | Import MD, DOCX, HTML as Google Docs |
| `import_to_google_slides` | Core | Import PPTX, PPT, ODP as Google Slides |
| `import_to_google_sheets` | Core | Import XLSX, CSV, TSV as Google Sheets |
| `get_drive_shareable_link` | Core | Get shareable links |
| `list_drive_items` | Extended | List folder contents, shared drives |
| `copy_drive_file` | Extended | Copy files (templates) |
| `update_drive_file` | Extended | Update metadata, move, trash |
| `manage_drive_access` | Extended | Grant, update, revoke permissions |
| `set_drive_file_permissions` | Extended | Link sharing and file-level permissions |
| `get_drive_file_permissions` | Complete | Detailed file permissions |
| `check_drive_file_public_access` | Complete | Verify public link sharing |

**OAuth scopes**: `drive.readonly`, `drive`, `drive.file`

## Google Calendar (7 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `list_calendars` | Core | List accessible calendars |
| `get_events` | Core | Query by time range, search, ID |
| `manage_event` | Core | Create, update, delete events |
| `create_calendar` | Extended | Create secondary calendar |
| `query_freebusy` | Extended | Free/busy queries |
| `manage_out_of_office` | Extended | OOO event CRUD |
| `manage_focus_time` | Extended | Focus time event CRUD |

**Event features**: Timezone support, transparency, visibility, reminders, Meet integration, attendees, attachments.

**OAuth scopes**: `calendar.readonly`, `calendar`, `calendar.events`

## Google Docs (20 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `get_doc_content` | Core | Extract document text |
| `create_doc` | Core | Create new documents |
| `modify_doc_text` | Core | Insert, replace, format text, links |
| `export_doc_to_pdf` | Extended | Export to PDF |
| `search_docs` | Extended | Find documents by name |
| `find_and_replace_doc` | Extended | Global find/replace |
| `list_docs_in_folder` | Extended | List docs in folder |
| `insert_doc_elements` | Extended | Tables, lists, page breaks |
| `update_paragraph_style` | Extended | Headings, lists, spacing, shading |
| `get_doc_as_markdown` | Extended | Export as Markdown |
| `list_document_comments` | Extended | List all comments |
| `manage_document_comment` | Extended | Create, reply, resolve |
| `insert_doc_image` | Complete | Insert images from Drive/URL |
| `update_doc_headers_footers` | Complete | Modify headers/footers |
| `batch_update_doc` | Complete | Atomic multi-step operations |
| `inspect_doc_structure` | Complete | Analyze structure for safe inserts |
| `create_table_with_data` | Complete | Create and populate tables |
| `debug_table_structure` | Complete | Debug cell positions |
| `manage_doc_tab` | Complete | Create, rename, delete tabs |

**OAuth scopes**: `documents.readonly`, `documents`

## Google Sheets (12 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `read_sheet_values` | Core | Read cell ranges |
| `modify_sheet_values` | Core | Write, update, clear cells |
| `create_spreadsheet` | Core | Create new spreadsheets |
| `list_spreadsheets` | Extended | List accessible spreadsheets |
| `get_spreadsheet_info` | Extended | Metadata, sheets, conditional formats |
| `format_sheet_range` | Extended | Colors, number formats, text styling |
| `list_sheet_tables` | Extended | Structured tables with IDs and ranges |
| `create_sheet` | Complete | Add sheets to existing files |
| `append_table_rows` | Complete | Append to structured table |
| `move_sheet_rows` | Complete | Move rows between sheets |
| `list_spreadsheet_comments` | Complete | List all comments |
| `manage_spreadsheet_comment` | Complete | Create, reply, resolve |
| `manage_conditional_formatting` | Complete | Add, update, delete rules |

**OAuth scopes**: `spreadsheets.readonly`, `spreadsheets`

## Google Slides (7 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `create_presentation` | Core | Create new presentations |
| `get_presentation` | Core | Presentation details + text extraction |
| `batch_update_presentation` | Extended | Create slides, shapes, text |
| `get_page` | Extended | Specific slide details |
| `get_page_thumbnail` | Extended | PNG thumbnails |
| `list_presentation_comments` | Complete | List all comments |
| `manage_presentation_comment` | Complete | Create, reply, resolve |

**OAuth scopes**: `presentations.readonly`, `presentations`

## Google Forms (6 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `create_form` | Core | Create forms |
| `get_form` | Core | Form details, questions, URLs |
| `list_form_responses` | Extended | List responses with pagination |
| `set_publish_settings` | Complete | Configure template/auth settings |
| `get_form_response` | Complete | Individual response details |
| `batch_update_form` | Complete | Batch updates to questions/items |

**OAuth scopes**: `forms.body`, `forms.body.readonly`, `forms.responses.readonly`

## Google Tasks (5 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `list_tasks` | Core | List with filtering, subtask hierarchy |
| `get_task` | Core | Task details |
| `manage_task` | Core | Create, update, delete, move |
| `list_task_lists` | Complete | All task lists |
| `get_task_list` | Complete | Task list details |
| `manage_task_list` | Complete | CRUD + clear completed |

**OAuth scopes**: `tasks.readonly`, `tasks`

## Google Contacts (8 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `search_contacts` | Core | Search by name, email, phone |
| `get_contact` | Core | Detailed contact info |
| `list_contacts` | Core | List with pagination |
| `manage_contact` | Core | Create, update, delete |
| `list_contact_groups` | Extended | Contact groups/labels |
| `get_contact_group` | Extended | Group details with members |
| `manage_contacts_batch` | Complete | Batch CRUD |
| `manage_contact_group` | Complete | CRUD + membership |

**OAuth scopes**: `contacts.readonly`, `contacts`

## Google Chat (5 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `send_message` | Core | Send to spaces |
| `get_messages` | Core | Retrieve space messages |
| `search_messages` | Core | Search chat history |
| `create_reaction` | Core | Add emoji reactions |
| `list_spaces` | Extended | List rooms and DMs |
| `download_chat_attachment` | Extended | Download attachments |

**OAuth scopes**: `chat.messages.readonly`, `chat.messages`, `chat.spaces.readonly`, `chat.spaces`

## Google Apps Script (13 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `list_script_projects` | Core | List accessible projects |
| `get_script_project` | Core | Complete project with files |
| `get_script_content` | Core | Specific file content |
| `create_script_project` | Core | Standalone or bound project |
| `update_script_content` | Core | Update or create files |
| `run_script_function` | Core | Execute with parameters |
| `generate_trigger_code` | Core | Generate trigger code |
| `manage_deployment` | Extended | Create, update, delete |
| `list_deployments` | Extended | All project deployments |
| `delete_script_project` | Extended | Delete projects |
| `list_versions` | Extended | List versions |
| `create_version` | Extended | Create version |
| `get_version` | Extended | Get version details |
| `list_script_processes` | Extended | Recent executions |
| `get_script_metrics` | Extended | Execution metrics |

**OAuth scopes**: `script.projects`, `script.deployments`, `script.processes`, `script.metrics`

## Google Custom Search (2 tools)

| Tool | Tier | Description |
|------|------|-------------|
| `search_custom` | Core | Web search with filters |
| `get_search_engine_info` | Complete | Engine metadata |

**Requires**: `GOOGLE_PSE_API_KEY` + `GOOGLE_PSE_ENGINE_ID`

## Auth Tools

| Tool | Tier | Description |
|------|------|-------------|
| `start_google_auth` | Complete | Legacy OAuth 2.0 flow (disabled in OAuth 2.1 mode) |

## Common Tool Patterns

- **Consolidated CRUD**: `manage_*` tools use an `action` parameter (`create`/`update`/`delete`)
- **Service injection**: All tools get authenticated Google API service objects via `@require_google_service()`
- **Email parameter**: `user_google_email` present on most tools (optional when `USER_GOOGLE_EMAIL` is set)
- **Batch operations**: Suffixed `_batch` for bulk reads/writes
