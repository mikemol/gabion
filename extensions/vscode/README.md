# Gabion VS Code Extension (Thin Wrapper)

This extension launches the Gabion Python LSP server over stdio and keeps
analysis logic on the server side.

## Project setup

1. Install and activate the repo toolchain:
   ```bash
   mise install
   mise exec -- python -m pip install -e .
   ```
2. In VS Code, choose the Python interpreter that has `gabion` installed:
   - Command Palette → `Python: Select Interpreter`
   - Pick the interpreter used by `mise exec -- python`.
3. Configure the extension if needed:
   - `gabion.pythonPath` (default `python`)
   - `gabion.serverArgs` (default `-m gabion.server`)

## What appears in the editor

- **Diagnostics** are published by the Gabion server (for example, implicit
  bundle detections).
- **Code actions** are surfaced for Gabion diagnostics:
  - `Gabion: Synthesis plan for bundle`
  - `Gabion: Refactor protocol from bundle`
- **Result panels** are emitted into the **Gabion** output channel for synthesis,
  refactor, and structure diff command responses.

## Live analysis progress

The extension remains a thin wrapper: it listens for server-side progress and
renders status in VS Code without running analysis logic in the client.

- The client subscribes to server `$/progress` notifications and reacts to the
  `gabion.dataflowAudit/progress-v1` token.
- During longer-running server analysis, VS Code shows a progress notification
  so users can see that work is still in flight.
- Progress notifications close when terminal markers are received (`done`,
  timeout classifications, and failure states), and also when request-failure
  cleanup runs.
- The **Gabion** output channel includes progress log lines in addition to
  command result payloads.

### Troubleshooting timeouts

If an analysis times out:

1. Open the **Gabion** output channel to inspect timeout context and the report
   artifact paths emitted by the server.
2. Review the referenced timeout report artifact(s) to confirm whether the
   timeout classification was expected for the current file/workspace size.
3. Retry the command after narrowing scope (for example, smaller target set or
   less competing editor activity) so the server can complete within budget.

## Command Palette actions

The extension wires these server commands directly:

- `Gabion: Synthesis Plan` → `gabion.synthesisPlan`
  - Prompted for bundle fields and existing protocol names.
  - Prints synthesis suggestions and warnings/errors in the output channel.
- `Gabion: Refactor Protocol` → `gabion.refactorProtocol`
  - Prompted for protocol name, bundle fields, and optional target functions.
  - Prints refactor entry-point edits and warnings/errors in the output channel.
- `Gabion: Structure Diff` → `gabion.structureDiff`
  - Prompted for baseline/current snapshot paths.
  - Prints diff results in the output channel.

## End-to-end example

1. Open a Python file containing a function with repeated parameter bundles.
2. Save the file and wait for Gabion diagnostics to appear.
3. Use `Quick Fix...` on a Gabion diagnostic and choose
   `Gabion: Synthesis plan for bundle`.
4. Review the generated protocol suggestions in the **Gabion** output channel.
5. Run `Gabion: Refactor Protocol` from the Command Palette and provide the
   proposed protocol name and bundle fields.
6. Review the returned edit plan in the output channel and apply it manually
   (or via your own workflow tooling).

## Development (local)

```bash
cd extensions/vscode
npm install
npm run check
```

Use VS Code **Run Extension** (F5) to launch a development host.
