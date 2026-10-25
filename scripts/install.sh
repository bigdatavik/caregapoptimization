#!/usr/bin/env bash
# One-command install of caregapoptimization into a Databricks workspace:
# login (if needed) → pre-checks → validate → deploy → run the job → start the app → print the app URL.
#
# Usage:
#   ./scripts/install.sh --host https://<workspace-url> --catalog <catalog> [options]
#   ./scripts/install.sh --profile <profile>              --catalog <catalog> [options]
#
# Options:
#   --schema <name>          Schema for tables + model            (default: caregapoptimization)
#   --app-name <name>        Databricks App name, unique          (default: caregapoptimization)
#   --endpoint-name <name>   Serving endpoint name, unique        (default: same as --app-name)
#   --deploy-endpoint        Also create a scale-to-zero serving endpoint (costs money when active)
#   --skip-job               Deploy and start the app, but don't run the 12-task job
#   --yes                    Don't ask for confirmation
#
# Settings are saved to .databricks/bundle/dev/variable-overrides.json, so later
# `databricks bundle ...` commands reuse them without --var flags.
set -euo pipefail

HOST="" PROFILE="" CATALOG="" SCHEMA="caregapoptimization" APP="caregapoptimization" ENDPOINT=""
DEPLOY_ENDPOINT="false" SKIP_JOB="false" YES="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --catalog) CATALOG="$2"; shift 2 ;;
    --schema) SCHEMA="$2"; shift 2 ;;
    --app-name) APP="$2"; shift 2 ;;
    --endpoint-name) ENDPOINT="$2"; shift 2 ;;
    --deploy-endpoint) DEPLOY_ENDPOINT="true"; shift ;;
    --skip-job) SKIP_JOB="true"; shift ;;
    --yes|-y) YES="true"; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)"; exit 2 ;;
  esac
done
ENDPOINT="${ENDPOINT:-$APP}"

step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*"; exit 1; }

cd "$(dirname "$0")/.."
[[ -f databricks.yml ]] || die "Run this from the caregapoptimization repo."
[[ -n "$CATALOG" ]] || die "--catalog is required (an existing Unity Catalog catalog you can create schemas in)."
[[ -n "$HOST" || -n "$PROFILE" ]] || die "Give --host https://<workspace-url> or --profile <name>."
[[ "$APP" =~ ^[a-z0-9-]{2,30}$ ]] || die "--app-name must be 2-30 chars: lowercase letters, numbers, hyphens."
[[ "$ENDPOINT" =~ ^[A-Za-z0-9_-]+$ ]] || die "--endpoint-name may only use letters, numbers, '-' and '_'."

step "1/8  Databricks CLI"
command -v databricks >/dev/null || die "Databricks CLI not found. Install: https://docs.databricks.com/dev-tools/cli/install.html"
VER="$(databricks --version | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
[[ "$(printf '%s\n0.250.0\n' "$VER" | sort -V | head -1)" == "0.250.0" ]] || die "CLI $VER is too old; need >= 0.250.0 (brew upgrade databricks)."
ok "CLI $VER"

step "2/8  Workspace login"
if [[ -n "$HOST" ]]; then
  [[ "$HOST" == http* ]] || HOST="https://$HOST"
  HOST="${HOST%/}"
  if [[ -z "$PROFILE" ]]; then
    PROFILE="$(databricks auth profiles -o json 2>/dev/null | python3 -c "
import json,sys
h=sys.argv[1].rstrip('/')
for p in json.load(sys.stdin).get('profiles',[]):
    if p.get('host','').rstrip('/')==h and p.get('valid'): print(p['name']); break" "$HOST" || true)"
  fi
  if [[ -z "$PROFILE" ]]; then
    PROFILE="caregap-$(echo "$HOST" | sed -E 's#https?://##; s#\..*##')"
    echo "  No working profile for $HOST; opening a browser to log in as profile '$PROFILE'…"
    databricks auth login --host "$HOST" --profile "$PROFILE"
  fi
fi
ME="$(databricks current-user me --profile "$PROFILE" -o json 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["userName"])')" \
  || die "Profile '$PROFILE' can't authenticate. Run: databricks auth login --host <workspace-url> --profile $PROFILE"
ok "Profile '$PROFILE' → $ME"

step "3/8  Pre-checks (catalog, Apps, app name)"
databricks catalogs get "$CATALOG" --profile "$PROFILE" >/dev/null 2>&1 || {
  echo "  Catalogs you can see:"; databricks catalogs list --profile "$PROFILE" -o json 2>/dev/null \
    | python3 -c 'import json,sys;[print("   -",c["name"]) for c in json.load(sys.stdin)]' | head -20
  die "Catalog '$CATALOG' not found or not accessible. Pick one above (you need USE CATALOG + CREATE SCHEMA)."; }
ok "Catalog '$CATALOG' is visible"
databricks apps list --profile "$PROFILE" >/dev/null 2>&1 || die "Databricks Apps isn't available to you in this workspace."
ok "Databricks Apps available"
if APPJSON="$(databricks apps get "$APP" --profile "$PROFILE" -o json 2>/dev/null)"; then
  echo "$APPJSON" | grep -q "Care-Gap Optimizer" \
    || die "An app named '$APP' already exists and isn't this project. Re-run with --app-name <another-name>."
  ok "App '$APP' exists from a previous install; it will be updated"
else
  ok "App name '$APP' is free"
fi

echo
echo "  Workspace : $(databricks auth describe --profile "$PROFILE" 2>/dev/null | grep -m1 -Eo 'https://[^ ]+' || echo "$HOST")"
echo "  Catalog   : $CATALOG    Schema: $SCHEMA"
echo "  App       : $APP    Endpoint: $ENDPOINT (deploy: $DEPLOY_ENDPOINT)    Run job: $([[ $SKIP_JOB == true ]] && echo no || echo yes)"
if [[ "$YES" != "true" ]]; then read -r -p "  Proceed? [y/N] " a; [[ "$a" =~ ^[Yy] ]] || die "Cancelled."; fi

step "4/8  Save settings (.databricks/bundle/dev/variable-overrides.json)"
mkdir -p .databricks/bundle/dev
cat > .databricks/bundle/dev/variable-overrides.json <<JSON
{
  "catalog": "$CATALOG",
  "schema": "$SCHEMA",
  "app_name": "$APP",
  "endpoint_name": "$ENDPOINT",
  "deploy_endpoint": "$DEPLOY_ENDPOINT"
}
JSON
ok "Saved; later 'databricks bundle …' commands reuse these"

step "5/8  Validate the bundle"
databricks bundle validate --strict -t dev --profile "$PROFILE" >/dev/null || databricks bundle validate --strict -t dev --profile "$PROFILE"
ok "Validation OK"

step "6/8  Deploy (job + app definition + files)"
databricks bundle deploy -t dev --profile "$PROFILE"
ok "Deployed"

if [[ "$SKIP_JOB" != "true" ]]; then
  step "7/8  Run the end-to-end job (data → model → optimizer → package → tests, ~10 min)"
  databricks bundle run caregapoptimization_e2e -t dev --profile "$PROFILE"
  ok "Job succeeded"
else
  step "7/8  Skipping the job (--skip-job)"
fi

step "8/8  Start the app"
databricks bundle run caregapoptimization_app -t dev --profile "$PROFILE"
URL="$(databricks apps get "$APP" --profile "$PROFILE" -o json | python3 -c 'import json,sys;print(json.load(sys.stdin).get("url",""))')"
ok "App running"

printf '\n\033[1;32mDone.\033[0m\n'
echo "  App:     $URL"
echo "  Tables:  $CATALOG.$SCHEMA  (members, gaps, intervention_effects, intervention_costs, vendor_capacities)"
echo "  Model:   $CATALOG.$SCHEMA.caregapoptimization_model"
echo "  Notebooks + job: databricks bundle summary -t dev --profile $PROFILE"
