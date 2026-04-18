# backend/data Directory Guide

`backend/data/` is a data workspace, not a code directory.

## Root-level rule

Keep `backend/data/` root clean:
- Keep only primary data directories and this README.
- Do not leave ad-hoc `*.html` / `*.json` debug artifacts directly in root.
- Put historical debug/evaluation artifacts under `backend/data/debug/`.

## Primary directories

- `raw/`: original source materials (syllabus/textbook/resource/hotspot).
- `processed/`: parser/normalization intermediate outputs.
- `structured/`: structured extraction outputs for Mongo ingestion.
- `chunks/`: chunking outputs used by vector ingestion.
- `relations/`: relation extraction artifacts.
- `graph/`: graph build artifacts and snapshots.
- `vector_index/`: vector index artifacts/reports.
- `hybrid_search/`: hybrid retrieval debug outputs.
- `ingest_reports/`: ingestion execution reports.
- `test_outputs/`: test run outputs.
- `test_tmp/`: temporary test workspace.
- `debug/`: non-production debug/evaluation artifacts.

## debug subdirectories

- `debug/raw_html/`: raw/rerun HTML pages captured during hotspot/debug workflows.
- `debug/eval_reports/`: eval/validation/verification reports.
- `debug/samples/`: sample datasets and sample extraction snapshots.
- `debug/pipeline_reports/`: historical ingest/from-artifacts/mongo-validation/extract-result style reports.

## Notes

- Reorganizing files in `backend/data/` should not change service/API behavior.
- Prefer moving uncertain historical artifacts into `debug/` instead of deleting them.
