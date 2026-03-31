#!/usr/bin/env bash

is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  is_sourced=1
fi

usage() {
  cat <<'EOF'
Usage: scripts/project-conda-env.sh [--dry-run] [--env-name NAME] [--env-file PATH]

Creates or updates the project conda environment using the repo folder name by default.

Source this script if you want it to activate the environment in your current shell:
  source scripts/project-conda-env.sh

Options:
  --dry-run          Print the action without changing anything
  --env-name NAME    Override the derived environment name
  --env-file PATH    Override the environment file path
  -h, --help         Show this help text
EOF
}

finish() {
  local status="${1:-0}"
  if [[ "$is_sourced" -eq 1 ]]; then
    return "$status"
  fi
  exit "$status"
}

main() {
  local dry_run=0
  local env_name=""
  local env_file=""
  local project_root=""
  local conda_env_names=""
  local verb=""
  local hook=""
  local -a cmd=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        dry_run=1
        ;;
      --env-name)
        shift
        env_name="${1:-}"
        ;;
      --env-file)
        shift
        env_file="${1:-}"
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        printf 'Unknown argument: %s\n\n' "$1" >&2
        usage >&2
        return 1
        ;;
    esac
    shift
  done

  if project_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    :
  else
    project_root="$(pwd)"
  fi

  if [[ -z "$env_name" ]]; then
    env_name="$(basename "$project_root")"
  fi

  env_name="${env_name// /-}"

  if [[ -z "$env_file" ]]; then
    env_file="$project_root/environment.yml"
  fi

  if [[ ! -f "$env_file" ]]; then
    printf 'Environment file not found: %s\n' "$env_file" >&2
    return 1
  fi

  if ! command -v conda >/dev/null 2>&1; then
    printf 'Could not find conda on PATH. Run this from a shell where conda is initialized.\n' >&2
    return 1
  fi

  if ! conda_env_names="$(conda env list | awk '!/^#/ && NF {print $1}')"; then
    printf 'Could not list conda environments.\n' >&2
    return 1
  fi

  if printf '%s\n' "$conda_env_names" | grep -Fxq "$env_name"; then
    verb="Updating"
    cmd=(conda env update -y -f "$env_file" -n "$env_name")
  else
    verb="Creating"
    cmd=(conda env create -y -f "$env_file" -n "$env_name")
  fi

  printf '%s conda environment: %s\n' "$verb" "$env_name"

  if [[ "$dry_run" -eq 1 ]]; then
    printf 'Dry run only. Environment file: %s\n' "$env_file"
    return 0
  fi

  if ! "${cmd[@]}"; then
    printf 'Conda command failed.\n' >&2
    return 1
  fi

  printf 'Done. Environment ready: %s\n' "$env_name"

  if [[ "$is_sourced" -eq 1 ]]; then
    if ! hook="$(conda shell.bash hook 2>/dev/null)"; then
      printf 'Could not initialize the conda shell hook.\n' >&2
      return 1
    fi

    eval "$hook"

    if ! conda activate "$env_name"; then
      printf 'Could not activate environment: %s\n' "$env_name" >&2
      return 1
    fi

    printf 'Activated: %s\n' "$env_name"
  else
    printf 'Activate with: conda activate %s\n' "$env_name"
  fi
}

main "$@"
finish "$?"
