# Gabion VS Code Extension (Thin Wrapper)

This is a thin wrapper that launches the Gabion Python LSP server over stdio.
It does not embed analysis logic.

## Requirements
- Python environment with `gabion` installed (editable or package).
- VS Code with Node available for extension dependencies.

## Configuration
- `gabion.pythonPath`: Python executable that can run the server.
- `gabion.serverArgs`: Arguments passed to the Python command (default: `-m gabion.server`).

## Commands
- `Gabion: Extract Protocol (stub)` runs the refactor protocol command against
  the active file and prints the response in the "Gabion" output channel.

## Development (local)
```
cd extensions/vscode
npm install
```

Use VS Code "Run Extension" (F5) to launch a dev instance.
