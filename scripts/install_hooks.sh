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

if [ -n "${GABION_SPPF_SYNC:-}" ]; then
  current_branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$current_branch" = "stage" ]; then
    "$repo_root/scripts/sppf_sync.py" --comment --label done-on-stage --range origin/stage..HEAD || true
  fi
fi
EOF

chmod +x "$hooks_dir/pre-commit" "$hooks_dir/pre-push"

echo "Installed git hooks: pre-commit, pre-push"
