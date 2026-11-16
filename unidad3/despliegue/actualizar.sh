#!/usr/bin/env bash
set -euo pipefail
# Instala únicamente el último commit de main cuya ejecución de CI terminó correctamente.
root=/home/azureuser/iaupeu-fraude
live="$root/releases/0c78ce2/unidad3/web"
repo="$root/repositorio.git"
exec 9>"$root/actualizar.lock"
flock -n 9 || exit 0
head_sha=$(git ls-remote https://github.com/glender222/iaupeu.git refs/heads/main | cut -f1)
[[ "$head_sha" =~ ^[0-9a-f]{40}$ ]]
run=$(curl -fsS --connect-timeout 15 --max-time 30 -H 'Accept: application/vnd.github+json' 'https://api.github.com/repos/glender222/iaupeu/actions/workflows/fraude-ci.yml/runs?branch=main&event=push&per_page=1')
ready=$(printf '%s' "$run" | python3 -c 'import json,sys; r=json.load(sys.stdin).get("workflow_runs", []); print(r[0]["head_sha"] if r and r[0]["status"]=="completed" and r[0]["conclusion"]=="success" else "")')
if [[ "$ready" != "$head_sha" ]]; then echo 'Último commit aún no validado; no se despliega.'; exit 0; fi
if [[ -f "$root/version-activa" ]] && [[ $(cat "$root/version-activa") == "$head_sha" ]]; then echo "Sin cambios: $head_sha"; exit 0; fi
if [[ ! -d "$repo" ]]; then git init --bare -q "$repo"; fi
git -C "$repo" fetch -q --depth=1 https://github.com/glender222/iaupeu.git "$head_sha"
release="$root/versiones/$head_sha"
if [[ ! -d "$release" ]]; then
  tmp=$(mktemp -d "$root/versiones/.preparar.XXXXXXXX")
  trap 'rm -rf "${tmp:-}"' EXIT
  git -C "$repo" archive FETCH_HEAD unidad3/web | tar -x -C "$tmp"
  test -s "$tmp/unidad3/web/index.html"
  test -s "$tmp/unidad3/web/modelos.json"
  mv "$tmp" "$release"
  trap - EXIT
fi
previous=$(readlink "$live")
test -n "$previous"
ln -s "$release/unidad3/web" "$live.nuevo"
mv -Tf "$live.nuevo" "$live"
if ! curl -fsS --connect-timeout 5 http://127.0.0.1:8000/fraude/ -o /dev/null || ! curl -fsS --connect-timeout 5 http://127.0.0.1:8000/fraude/modelos.json -o /dev/null || ! curl -fsS --connect-timeout 5 http://127.0.0.1:8000/health -o /dev/null; then
  ln -s "$previous" "$live.restaurar"
  mv -Tf "$live.restaurar" "$live"
  echo 'Falló la comprobación; se restauró la versión anterior.' >&2
  exit 1
fi
printf '%s\n' "$head_sha" > "$root/version-activa.tmp"
mv -f "$root/version-activa.tmp" "$root/version-activa"
echo "Desplegado: $head_sha"
