"use strict";

const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const config = vscode.workspace.getConfiguration("gabion");
  const command = config.get("pythonPath", "python");
  const args = config.get("serverArgs", ["-m", "gabion.server"]);

  const serverOptions = {
    command,
    args,
    transport: TransportKind.stdio,
  };

  const outputChannel = vscode.window.createOutputChannel("Gabion");
  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "python" }],
    outputChannel,
  };

  client = new LanguageClient("gabion", "Gabion LSP", serverOptions, clientOptions);
  context.subscriptions.push(client.start(), outputChannel);

  const refactorCommand = vscode.commands.registerCommand(
    "gabion.refactorProtocol",
    async () => {
      await client.onReady();
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("Gabion: no active editor found.");
        return;
      }
      const protocolName =
        (await vscode.window.showInputBox({
          prompt: "Protocol name",
          value: "TODO_Bundle",
        })) || "TODO_Bundle";
      const bundleInput = await vscode.window.showInputBox({
        prompt: "Bundle fields (comma-separated)",
        value: "",
      });
      const bundle = bundleInput
        ? bundleInput
            .split(",")
            .map((item) => item.trim())
            .filter((item) => item.length > 0)
        : [];
      const payload = {
        protocol_name: protocolName,
        bundle,
        target_path: editor.document.uri.fsPath,
        target_functions: [],
        rationale: "Manual refactor protocol command (stub).",
      };
      try {
        const result = await client.sendRequest("workspace/executeCommand", {
          command: "gabion.refactorProtocol",
          arguments: [payload],
        });
        outputChannel.appendLine(
          `Refactor response: ${JSON.stringify(result, null, 2)}`
        );
        outputChannel.show(true);
      } catch (err) {
        vscode.window.showErrorMessage(`Gabion refactor failed: ${err}`);
      }
    }
  );
  context.subscriptions.push(refactorCommand);
}

function deactivate() {
  if (!client) {
    return undefined;
  }
  return client.stop();
}

module.exports = { activate, deactivate };
