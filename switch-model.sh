#!/usr/bin/env bash
# One-command model switch on the server. Validates the profile against
# profiles.py, stops whatever instance is running (single GPU -> only one at a
# time), then starts the requested one.
#
#   sudo ./switch-model.sh gemma
#
# Equivalent to:
#   sudo systemctl stop 'platform-ai-service@*'
#   sudo systemctl start platform-ai-service@gemma
set -euo pipefail

key="${1:?usage: switch-model <profile>   (see: bin/python profiles.py list)}"
repo="$(cd "$(dirname "$0")" && pwd)"

# Fail before touching the running service if the key is unknown.
"$repo/bin/python" "$repo/profiles.py" validate "$key"

systemctl stop 'platform-ai-service@*' 2>/dev/null || true
systemctl start "platform-ai-service@${key}"
systemctl --no-pager status "platform-ai-service@${key}" | head -n 5
