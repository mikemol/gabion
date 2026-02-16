"use strict";

const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

const SYNTHESIS_COMMAND = "gabion.synthesisPlan";
const REFACTOR_COMMAND = "gabion.refactorProtocol";
const STRUCTURE_DIFF_COMMAND = "gabion.structureDiff";

let client;

function createResultPanel(outputChannel, title, payload, result) {
  outputChannel.appendLine(`\\n=== ${title} ===`);
  outputChannel.appendLine(`Payload:`);
  outputChannel.appendLine(JSON.stringify(payload, null, 2));
  outputChannel.appendLine("Result:");
  outputChannel.appendLine(JSON.stringify(result, null, 2));
  outputChannel.show(true);
}

async function executeServerCommand({ command, payload, outputChannel, responseTitle }) {
  try {
    const result = await client.sendRequest("workspace/executeCommand", {
      command,
      arguments: [payload],
    });
    createResultPanel(outputChannel, responseTitle, payload, result);
    return result;
  } catch (err) {
    vscode.window.showErrorMessage(`Gabion command failed (${command}): ${err}`);
    throw err;
  }
}

function parseCsv(input) {
  return (input || "")
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function normalizeLineInput(value, fallback) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    return fallback;
  }
  return parsed;
}

async function promptForSynthesisPayload(editor) {
  const bundleInput = await vscode.window.showInputBox({
    prompt: "Bundle fields (comma-separated)",
    value: "",
  });
  if (bundleInput === undefined) {
    return null;
  }

  const existingNamesInput = await vscode.window.showInputBox({
    prompt: "Existing protocol names (comma-separated, optional)",
    value: "",
  });
  if (existingNamesInput === undefined) {
    return null;
  }

  return {
    bundles: [{ bundle: parseCsv(bundleInput), tier: 2 }],
    existing_names: parseCsv(existingNamesInput),
    field_types: {},
    fallback_prefix: "GabionBundle",
    max_tier: 3,
    min_bundle_size: 2,
    allow_singletons: false,
    merge_overlap_threshold: 0.5,
    source_path: editor?.document.uri.fsPath || "",
  };
}

async function promptForRefactorPayload(editor, seedBundle = []) {
  const protocolNameInput = await vscode.window.showInputBox({
    prompt: "Protocol name",
    value: "ExtractedBundle",
  });
  if (protocolNameInput === undefined) {
    return null;
  }

  const bundleInput = await vscode.window.showInputBox({
    prompt: "Bundle fields (comma-separated)",
    value: seedBundle.join(", "),
  });
  if (bundleInput === undefined) {
    return null;
  }

  const targetFunctionsInput = await vscode.window.showInputBox({
    prompt: "Target function names (comma-separated, optional)",
    value: "",
  });
  if (targetFunctionsInput === undefined) {
    return null;
  }

  return {
    protocol_name: protocolNameInput || "ExtractedBundle",
    bundle: parseCsv(bundleInput),
    fields: [],
    target_path: editor?.document.uri.fsPath || "",
    target_functions: parseCsv(targetFunctionsInput),
    compatibility_shim: false,
    rationale: "Manual refactor protocol command from VS Code extension.",
  };
}

async function promptForStructureDiffPayload() {
  const baselinePath = await vscode.window.showInputBox({
    prompt: "Baseline snapshot path",
    value: "artifacts/structure/baseline.json",
  });
  if (!baselinePath) {
    return null;
  }

  const currentPath = await vscode.window.showInputBox({
    prompt: "Current snapshot path",
    value: "artifacts/structure/current.json",
  });
  if (!currentPath) {
    return null;
  }

  return { baseline: baselinePath, current: currentPath };
}

function bundleFromDiagnostic(diagnostic) {
  const marker = "Implicit bundle detected:";
  if (!diagnostic || typeof diagnostic.message !== "string") {
    return [];
  }
  const idx = diagnostic.message.indexOf(marker);
  if (idx < 0) {
    return [];
  }
  return parseCsv(diagnostic.message.slice(idx + marker.length));
}

function createCodeActionsForDiagnostic(diagnostic) {
  const bundle = bundleFromDiagnostic(diagnostic);
  const range = diagnostic.range;

  const synthesize = new vscode.CodeAction(
    "Gabion: Synthesis plan for bundle",
    vscode.CodeActionKind.QuickFix
  );
  synthesize.command = {
    title: "Gabion: Synthesis plan",
    command: SYNTHESIS_COMMAND,
    arguments: [{ bundle, range }],
  };
  synthesize.diagnostics = [diagnostic];

  const refactor = new vscode.CodeAction(
    "Gabion: Refactor protocol from bundle",
    vscode.CodeActionKind.RefactorExtract
  );
  refactor.command = {
    title: "Gabion: Refactor protocol",
    command: REFACTOR_COMMAND,
    arguments: [{ bundle, range }],
  };
  refactor.diagnostics = [diagnostic];

  return [synthesize, refactor];
}

function registerGabionCommands(context, outputChannel) {
  const synthesisCommand = vscode.commands.registerCommand(
    SYNTHESIS_COMMAND,
    async (seed = {}) => {
      await client.onReady();
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("Gabion: no active editor found.");
        return;
      }
      const payload = await promptForSynthesisPayload(editor);
      if (!payload) {
        return;
      }
      if (Array.isArray(seed.bundle) && seed.bundle.length > 0) {
        payload.bundles = [{ bundle: seed.bundle, tier: 2 }];
      }
      await executeServerCommand({
        command: SYNTHESIS_COMMAND,
        payload,
        outputChannel,
        responseTitle: "Synthesis Suggestions",
      });
    }
  );

  const refactorCommand = vscode.commands.registerCommand(
    REFACTOR_COMMAND,
    async (seed = {}) => {
      await client.onReady();
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("Gabion: no active editor found.");
        return;
      }

      const payload = await promptForRefactorPayload(
        editor,
        Array.isArray(seed.bundle) ? seed.bundle : []
      );
      if (!payload) {
        return;
      }

      if (seed.range && editor.selection.isEmpty) {
        const startLine = normalizeLineInput(seed.range.start?.line + 1, 1);
        const endLine = normalizeLineInput(seed.range.end?.line + 1, startLine);
        payload.target_functions = payload.target_functions.length
          ? payload.target_functions
          : [`lines:${startLine}-${endLine}`];
      }

      await executeServerCommand({
        command: REFACTOR_COMMAND,
        payload,
        outputChannel,
        responseTitle: "Refactor Entry Points",
      });
    }
  );

  const structureDiffCommand = vscode.commands.registerCommand(
    STRUCTURE_DIFF_COMMAND,
    async () => {
      await client.onReady();
      const payload = await promptForStructureDiffPayload();
      if (!payload) {
        return;
      }
      await executeServerCommand({
        command: STRUCTURE_DIFF_COMMAND,
        payload,
        outputChannel,
        responseTitle: "Structure Diff",
      });
    }
  );

  context.subscriptions.push(synthesisCommand, refactorCommand, structureDiffCommand);
}

function registerCodeActionProvider(context) {
  const provider = {
    provideCodeActions(_document, _range, codeActionContext) {
      const actions = [];
      for (const diagnostic of codeActionContext.diagnostics) {
        if (diagnostic?.source !== "gabion") {
          continue;
        }
        actions.push(...createCodeActionsForDiagnostic(diagnostic));
      }
      return actions;
    },
  };

  const registration = vscode.languages.registerCodeActionsProvider(
    { scheme: "file", language: "python" },
    provider,
    { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix, vscode.CodeActionKind.RefactorExtract] }
  );
  context.subscriptions.push(registration);
}

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

  registerGabionCommands(context, outputChannel);
  registerCodeActionProvider(context);
}

function deactivate() {
  if (!client) {
    return undefined;
  }
  return client.stop();
}

module.exports = {
  activate,
  deactivate,
  __test__: {
    bundleFromDiagnostic,
    createCodeActionsForDiagnostic,
    parseCsv,
  },
};
