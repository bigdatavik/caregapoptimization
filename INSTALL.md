# Install caregapoptimization in your Databricks workspace

Everything installs through one Declarative Automation Bundle (`databricks.yml`). You don't edit any code:
you only choose a workspace, a catalog, and (optionally) names.

At the end you'll have:

| What | Where |
|---|---|
| 5 synthetic tables | `<catalog>.<schema>`: members, gaps, intervention_effects, intervention_costs, vendor_capacities |
| A registered model (the whole engine) | `<catalog>.<schema>.caregapoptimization_model` |
| An MLflow experiment | `/Users/<you>/caregapoptimization_experiments` |
| A 12-task job | `[dev <you>] caregapoptimization_e2e` (notebooks 00–11, serverless) |
| The app | Databricks App `<app_name>` (Configure & Optimize, Scenario Lab, Member Explorer, Data Quality, Feedback, About) |
| Optional | A scale-to-zero serving endpoint (`--deploy-endpoint`) |

All data is synthetic. Nothing reads real member data.

---

## 1. Before you start

**On your laptop**
- Databricks CLI **0.250.0 or newer**: `databricks --version` (install / upgrade: `brew install databricks` or
  `brew upgrade databricks`; other platforms: https://docs.databricks.com/dev-tools/cli/install.html).
- `git` and `python3` (used by the installer for small JSON lookups).
- This repo: `git clone https://github.com/bigdatavik/caregapoptimization.git && cd caregapoptimization`

**In the workspace**
- Unity Catalog enabled, with **serverless compute for jobs and notebooks**.
- **Databricks Apps** enabled.
- An existing catalog where you have **USE CATALOG** and **CREATE SCHEMA**. The job creates the schema,
  the tables (CREATE TABLE) and the model (CREATE MODEL) inside it. If you'd rather use a dedicated catalog,
  you need CREATE CATALOG on the metastore; most users don't have it, so pick an existing one.
- Permission to create jobs and apps (the default for workspace users).

Ask your workspace admin if any of these are missing. A quick check of the catalog:
`databricks catalogs get <catalog> --profile <PROFILE>`.

---

## 2. Install with one command (recommended)

```bash
./scripts/install.sh --host https://<your-workspace-url> --catalog <catalog>
```

That's it. The script:
1. checks the CLI version;
2. finds a working CLI profile for that host, or opens a browser to log you in (OAuth) and saves a profile;
3. checks that the catalog exists and is visible to you, that Databricks Apps is available, and that the app name is free;
4. saves your settings to `.databricks/bundle/dev/variable-overrides.json` (so later commands can't silently fall back to defaults);
5. validates the bundle (`bundle validate --strict`);
6. deploys the job, the app definition and the files (`bundle deploy`);
7. runs the end-to-end job and waits for it (~10 minutes the first time; each notebook installs its pinned packages);
8. starts the app and prints its URL.

Already have a CLI profile? Use it instead of `--host`:

```bash
./scripts/install.sh --profile <PROFILE> --catalog <catalog>
```

### Options

| Option | Default | When to change it |
|---|---|---|
| `--schema` | `caregapoptimization` | You want a different schema (created if missing) |
| `--app-name` | `caregapoptimization` | **Someone else already installed it in this workspace.** App names are workspace-unique: 2–30 lowercase letters, numbers, hyphens |
| `--endpoint-name` | same as `--app-name` | Only matters with `--deploy-endpoint`; endpoint names are workspace-unique |
| `--deploy-endpoint` | off | You want a REST endpoint for the model. It scales to zero, but **costs money while active** |
| `--skip-job` | off | Deploy and start the app only (the app has its own in-memory synthetic data, so it works without the job) |
| `--yes` | off | Skip the confirmation prompt (CI / scripted installs) |

Example for a second install in the same workspace:

```bash
./scripts/install.sh --host https://<workspace-url> --catalog main --schema caregap_team2 --app-name caregap-team2
```

---

## 3. Install manually (the same steps, by hand)

```bash
# 1. Log in once and pick a profile name
databricks auth login --host https://<your-workspace-url> --profile <PROFILE>

# 2. Validate, then deploy. Pass the SAME --var flags to every bundle command.
databricks bundle validate --strict -t dev --profile <PROFILE> --var="catalog=<catalog>"
databricks bundle deploy          -t dev --profile <PROFILE> --var="catalog=<catalog>"

# 3. Run notebooks 00–11 end to end (~10 min)
databricks bundle run caregapoptimization_e2e -t dev --profile <PROFILE> --var="catalog=<catalog>"

# 4. Start the app, then get its URL
databricks bundle run caregapoptimization_app -t dev --profile <PROFILE> --var="catalog=<catalog>"
databricks apps get <app_name> --profile <PROFILE> -o json | python3 -c 'import json,sys;print(json.load(sys.stdin)["url"])'
```

Bundle variables (one `--var="key=value"` flag each):

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `main` | Catalog for the tables and the model |
| `schema` | `caregapoptimization` | Schema (created if missing) |
| `app_name` | `caregapoptimization` | App name (workspace-unique) |
| `endpoint_name` | `caregapoptimization` | Serving endpoint name used when `deploy_endpoint=true` |
| `deploy_endpoint` | `false` | `true` makes notebook 09 create / update a scale-to-zero endpoint |

**Tip:** instead of repeating `--var` flags, put them in `.databricks/bundle/dev/variable-overrides.json`
(that's what the installer does). The `.databricks/` folder is gitignored.

```json
{ "catalog": "<catalog>", "schema": "caregapoptimization", "app_name": "caregapoptimization",
  "endpoint_name": "caregapoptimization", "deploy_endpoint": "false" }
```

---

## 4. Check it worked

- **Job:** `databricks bundle summary -t dev --profile <PROFILE>` lists the job and app with links.
  All 12 tasks should be green; task 11 runs the 57 tests and fails the job if any fail.
- **Tables:** `databricks tables list <catalog> <schema> --profile <PROFILE>` shows 5 tables
  (members 500 rows, gaps 1,007).
- **Model:** Catalog Explorer → `<catalog>.<schema>.caregapoptimization_model` → version 1 READY.
- **App:** open the URL, go to **Configure & Optimize**, leave Balanced / $13,200 and click **Run Optimizer**.
  Expect: optimal · objective 160.8 · $13,156 / $13,200 · 857 / 3432 selected · 0 violations (~15 s).

---

## 5. Use it

- **Notebooks:** open them in the deployed bundle folder,
  `/Workspace/Users/<you>/.bundle/caregapoptimization/dev/files/notebooks`. Set the `catalog` and `schema`
  widgets at the top of a notebook to run it interactively; no code edits.
- **Recorded outputs:** Workflows → `caregapoptimization_e2e` → a run → click a task to see every cell's output.
- **Short demo:** notebook 05 runs the core flow end to end (data → baseline → candidates → optimizer → validation).

---

## 6. Update, re-run, remove

| Task | Command |
|---|---|
| Push code changes | `databricks bundle deploy -t dev --profile <PROFILE>` (re-run `install.sh` with the same options, or reuse the saved overrides) |
| Re-run the pipeline | `databricks bundle run caregapoptimization_e2e -t dev --profile <PROFILE>`. Safe to repeat: data generation is skipped if tables exist; each run adds a model version |
| Restart the app | `databricks bundle run caregapoptimization_app -t dev --profile <PROFILE>` |
| Regenerate the data | Drop the 5 tables (or the schema), then re-run the job |
| Remove the job, app and files | `databricks bundle destroy -t dev --profile <PROFILE>` |
| Remove data, model, endpoint | Not owned by the bundle. Delete the registered model (Catalog Explorer → model → Delete), then `DROP SCHEMA <catalog>.<schema> CASCADE` for the tables; delete the serving endpoint and the MLflow experiment if you created them |

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Catalog '<x>' not found or not accessible` | Pick a catalog from the list the installer prints, or ask an admin for USE CATALOG + CREATE SCHEMA |
| Job fails in task 02 with a catalog / schema error | You lack CREATE SCHEMA or CREATE TABLE on the catalog; use another catalog |
| `An app named '<x>' already exists` | Someone else installed it; re-run with `--app-name <new-name>` |
| `Databricks Apps isn't available` | Apps isn't enabled for the workspace or for you; ask an admin. You can still run the job with the manual steps |
| Profile can't authenticate / token expired | `databricks auth login --host <workspace-url> --profile <PROFILE>` |
| CLI too old | `brew upgrade databricks` (need >= 0.250.0) |
| A redeploy points at the wrong catalog | You ran a bundle command without your `--var` flags. Use the installer or `variable-overrides.json` |
| First job run is slow | Expected: each notebook `%pip install`s pyomo, highspy and lightgbm on serverless. Later runs are faster |
| Endpoint name conflict (with `--deploy-endpoint`) | Use `--endpoint-name <unique-name>` |
| App shows the default catalog on "About & Readiness" | Known: that page is mostly static. The app runs on its own synthetic data and doesn't read your catalog |

---

## 8. Going beyond dev

The bundle ships one target, `dev` (`mode: development`: resources are prefixed with `[dev <you>]`
and deployed under your user folder). For a shared install, add a `prod` target in `databricks.yml` with
`mode: production`, a shared `root_path`, and `run_as` / `permissions` for a service principal, then
`databricks bundle deploy -t prod`. See https://docs.databricks.com/dev-tools/bundles/deployment-modes.html.
