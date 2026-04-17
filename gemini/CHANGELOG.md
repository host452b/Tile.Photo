# CHANGELOG

> Convention: entries are YAML with fields date(ISO)/type/target/change/rationale/action/result/validation/status.
> `try-failed` entries are never deleted, only compressed. See `docs/superpowers/plans/2026-04-17-photomosaic-toy.md` for full convention.

## 活跃条目

- date: 2026-04-17
  type: feat
  target: repo
  change: Initial scaffolding (requirements, gitignore, module skeletons, plan doc)
  rationale: Kick off photomosaic toy per user spec (local ipynb + fun features + DeepZoom output)
  action: Create requirements.txt, .gitignore, README.md, src/tests skeletons
  result: Repo has no code yet, only structure
  validation: Tree inspection + pytest collects zero tests successfully
  status: stable
