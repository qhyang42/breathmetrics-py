## Quickstart
Airflow data only.
```bash
pip install breathmetrics
breathmetrics estimate examples/human_airflow.csv --fs 1000 --datatype humanAirflow --out results/
breathmetrics inspect results/bm.pkl
breathmetrics info results/bm.pkl
```

## 🛠️ Developer Shortcuts

Sometimes you just want to move fast and push messy code. Here are a few useful tricks:

### Skip pre-commit checks for one commit
```bash
git commit -m "WIP messy commit" --no-verify

### temporarilly disable hooks 
pre-commit uninstall
pre-commit install
pre-commit run --all-files

### run tests 
pytest -q 
