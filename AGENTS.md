# Repository development rules

- Keep test scripts, test fixtures, benchmarks created for validation, and generated test artifacts local. Do not commit or push them.
- Place temporary validation scripts and outputs under the ignored `tmp/` directory. `ml_alpha/tests/` must remain untracked.
- Do not write caches to the C: drive.
