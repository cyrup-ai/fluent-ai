# Agent Client Protocol (ACP)

## Overview

ACP standardizes JSON-RPC 2.0 communication over stdio between clients (e.g., code editors) and agents (AI-driven code modifiers). Agents run as client subprocesses, supporting concurrent sessions. Messages include methods (request-response) and notifications (one-way). User-readable text defaults to Markdown. Content blocks reuse MCP types for interoperability.

Protocol flow:
- Client initializes connection, negotiates version and capabilities.
- Client creates or loads sessions with working directory (absolute path) and MCP servers.
- Client sends prompts; agent processes, reports updates via notifications, and responds with stop reason.
- Agent may request permissions or file access if client supports.

All file paths are absolute; line numbers 1-based. Errors follow JSON-RPC 2.0.

## Initialization

Client calls `initialize` with supported protocol version (major integer) and capabilities.

Request params:
- `protocolVersion`: integer (required, latest supported).
- `clientCapabilities`: object (fs: {readTextFile: bool, writeTextFile: bool}).

Agent responds with:
- `protocolVersion`: integer (matched or latest supported; client closes if unsupported).
- `agentCapabilities`: object (loadSession: bool [default false], promptCapabilities: {image: bool, audio: bool, embeddedContext: bool} [all default false]).
- `authMethods`: array (if authentication required).

Baseline: All agents support text and resource_link in prompts. Capabilities enable optional features without breaking changes.

If authentication needed, agent lists methods; client calls `authenticate` with methodId.

## Session Setup

Sessions isolate conversations. Client provides cwd (absolute path) and mcpServers array (each: {name: string, command: string, args: array, env: array of {name, value}}).

### Create Session
Method: `session/new`.
Params: {cwd, mcpServers}.
Response: {sessionId: string}.

### Load Session
If agent supports loadSession, method: `session/load`.
Params: {sessionId, cwd, mcpServers}.
Agent streams history via `session/update` notifications (user_message_chunk, agent_message_chunk), then responds null.

## Prompt Turn Lifecycle

A turn starts with `session/prompt`, ends with response containing stopReason (end_turn, max_tokens, max_turn_requests, refusal, cancelled).

Method: `session/prompt`.
Params: {sessionId, prompt: array of ContentBlock (types per capabilities)}.
Response: {stopReason}.

During processing:
- Agent sends `session/update` notifications: {sessionId, update: union}.
  - plan: {entries: array of {content: string, priority: high|medium|low, status: pending|in_progress|completed}}.
  - agent_message_chunk: {content: ContentBlock}.
  - tool_call: {toolCallId: string, title: string, kind: read|edit|delete|move|search|execute|think|fetch|other, status: pending|in_progress|completed|failed, content: array of ToolCallContent, locations: array of {path: string, line: integer}, rawInput/rawOutput: object}.
  - tool_call_update: {toolCallId, [optional fields from tool_call]}.

If tool requires permission, agent calls `session/request_permission`.
Params: {sessionId, toolCall: ToolCallUpdate, options: array of {optionId: string, name: string, kind: allow_once|allow_always|reject_once|reject_always}}.
Response: {outcome: {outcome: selected|cancelled, [optionId]}}.

Client may send `session/cancel` notification: {sessionId}. Agent aborts, responds to prompt with cancelled.

Loop until no pending tools or stop.

## Content Blocks

Union types (compatible with MCP):
- text: {text: string}.
- image: {data: base64 string, mimeType: string} (if image capable).
- audio: {data: base64 string, mimeType: string} (if audio capable).
- resource_link: {uri: string, name: string, [mimeType, title, description, size]}.
- resource: {resource: union of {uri, text, mimeType} or {uri, blob: base64, mimeType}} (if embeddedContext capable).

All include optional annotations object.

## Tool Calls

Reported/updated via session/update.
ToolCallContent union:
- content: {content: ContentBlock}.
- diff: {path: string, oldText: string|null, newText: string}.

## File System Access

If client supports:
- `fs/read_text_file`: Params {sessionId, path, [line, limit]}. Response {content: string}.
- `fs/write_text_file`: Params {sessionId, path, content}. Response null.

## Schema Summary

### Types
- SessionId: string.
- ProtocolVersion: uint16.
- StopReason: enum (end_turn, max_tokens, max_turn_requests, refusal, cancelled).
- ToolCallStatus: enum (pending, in_progress, completed, failed).
- ToolKind: enum (read, edit, delete, move, search, execute, think, fetch, other).
- PermissionOptionKind: enum (allow_once, allow_always, reject_once, reject_always).

### Capabilities
- Agent: loadSession, promptCapabilities {image, audio, embeddedContext}.
- Client: fs {readTextFile, writeTextFile}.

### Methods (Agent)
- initialize: Params {protocolVersion, clientCapabilities}. Response {protocolVersion, agentCapabilities, authMethods}.
- authenticate: Params {methodId}. Response null.
- session/new: Params {cwd, mcpServers}. Response {sessionId}.
- session/load: Params {sessionId, cwd, mcpServers}. Response null.
- session/prompt: Params {sessionId, prompt}. Response {stopReason}.

### Notifications (Agent)
- session/cancel: Params {sessionId}.

### Methods (Client)
- session/request_permission: Params {sessionId, toolCall, options}. Response {outcome}.
- fs/read_text_file: Params {sessionId, path, [line, limit]}. Response {content}.
- fs/write_text_file: Params {sessionId, path, content}. Response null.

### Notifications (Client)
- session/update: Params {sessionId, update}.
