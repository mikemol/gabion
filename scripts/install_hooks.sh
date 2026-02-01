#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
hooks_dir="$repo_root/.git/hooks"

if [ ! -d "$hooks_dir" ]; then
  echo "No .git/hooks directory found. Are you inside a git repo?" >&2
  exit 1
fi

cat > "$hooks_dir/pre-commit" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ -n "${GABION_SKIP_HOOKS:-}" ]; then
  echo "GABION_SKIP_HOOKS set; skipping pre-commit checks." >&2
  exit 0
fi

repo_root=$(git rev-parse --show-toplevel)
"$repo_root/scripts/checks.sh" --dataflow-only
EOF

cat > "$hooks_dir/pre-push" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ -n "${GABION_SKIP_HOOKS:-}" ]; then
  echo "GABION_SKIP_HOOKS set; skipping pre-push checks." >&2
  exit 0
fi

repo_root=$(git rev-parse --show-toplevel)
"$repo_root/scripts/checks.sh" --no-docflow
EOF

chmod +x "$hooks_dir/pre-commit" "$hooks_dir/pre-push"

echo "Installed git hooks: pre-commit, pre-push"
