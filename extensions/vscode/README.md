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
