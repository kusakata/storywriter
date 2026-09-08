#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$DIR"
unset ELECTRON_RUN_AS_NODE

ELECTRON="$DIR/node_modules/electron/dist/electron"
if [[ ! -x "$ELECTRON" ]]; then
  msg="Electron が見つかりません。app ディレクトリで npm install してください。"
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="Storywriter" --text="$msg"
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "Storywriter" "$msg"
  else
    echo "$msg" >&2
  fi
  exit 1
fi

exec "$ELECTRON" "$DIR" --class=Storywriter
