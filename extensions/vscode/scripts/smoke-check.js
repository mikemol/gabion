"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");

const packageJsonPath = path.join(__dirname, "..", "package.json");
const extensionPath = path.join(__dirname, "..", "extension.js");

const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
const contributedCommands = new Set(
  (packageJson.contributes?.commands || []).map((entry) => entry.command)
);

for (const expected of [
  "gabion.synthesisPlan",
  "gabion.refactorProtocol",
  "gabion.structureDiff",
]) {
  assert(
    contributedCommands.has(expected),
    `Expected package.json to contribute command: ${expected}`
  );
  assert(
    (packageJson.activationEvents || []).includes(`onCommand:${expected}`),
    `Expected activation event for command: ${expected}`
  );
}

const originalLoad = Module._load;
Module._load = function patchedLoad(request, parent, isMain) {
  if (request === "vscode") {
    class CodeAction {
      constructor(title, kind) {
        this.title = title;
        this.kind = kind;
      }
    }
    return {
      CodeAction,
      CodeActionKind: {
        QuickFix: "quickfix",
        RefactorExtract: "refactor.extract",
      },
    };
  }
  if (request === "vscode-languageclient/node") {
    return {
      LanguageClient: function noop() {},
      TransportKind: { stdio: "stdio" },
    };
  }
  return originalLoad(request, parent, isMain);
};

const extensionModule = require(extensionPath);
const testApi = extensionModule.__test__;
assert(testApi, "Expected extension __test__ API exports");

const actions = testApi.createCodeActionsForDiagnostic({
  source: "gabion",
  message: "Implicit bundle detected: db, cache, logger",
  range: {
    start: { line: 3, character: 2 },
    end: { line: 3, character: 14 },
  },
});

assert.equal(actions.length, 2, "Expected synthesis/refactor code actions");
assert.equal(actions[0].command.command, "gabion.synthesisPlan");
assert.equal(actions[1].command.command, "gabion.refactorProtocol");
assert.deepEqual(actions[0].command.arguments[0].bundle, ["db", "cache", "logger"]);

console.log("VS Code extension smoke checks passed.");
