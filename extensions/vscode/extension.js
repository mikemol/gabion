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
}

function deactivate() {
  if (!client) {
    return undefined;
  }
  return client.stop();
}

module.exports = { activate, deactivate };
