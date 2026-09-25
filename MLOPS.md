# MLflow and DVC migration

Install `.[mlops]`, start `/home/devops/mlops`, and load its `.env` before
running `scripts/train_marian.py`. Each run is recorded in
`ukrainian-marian-verbalizer` with step-level losses/evaluations, the existing
run manifest, final reports, and the selected `final` checkpoint.

Inventory first and add only reviewed immutable datasets or release artifacts
to DVC. This non-Git project uses:

```bash
/home/devops/mlops/scripts/init-project-dvc.sh "$PWD" no-scm
```

No historical files are removed, relocated, or bulk-uploaded automatically.
