"""Run the four current Top50 research cases and save their exact configs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]
IDS = ['strategy_001_top50_5d', 'strategy_001_top50_exclude10_5d',
       'strategy_001_top50_exclude20_5d', 'strategy_001_top50_no_limits_5d']


def main():
    project = tomllib.loads((ROOT / 'config.toml').read_text(encoding='utf-8'))
    data = Path(project['paths']['base_data_dir'])
    if not data.is_absolute():
        data = ROOT / data
    executable = ROOT / 'factor_engine/target/release/yq-factor-engine.exe'
    for sid in IDS:
        cfg = ROOT / f'strategy_config/stock/{sid}.toml'
        run = subprocess.run([str(executable), 'strategy-run', '--config', str(cfg)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        out = data / 'strategy/stock' / sid
        out.mkdir(parents=True, exist_ok=True)
        (out / 'run.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        run.check_returncode()
        shutil.copy2(cfg, out / 'run_config.toml')
        (out / 'provenance.json').write_text(json.dumps({
            'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256(cfg.read_bytes()).hexdigest(),
            'holdings_sha256': hashlib.sha256((out / 'holdings.parquet').read_bytes()).hexdigest(),
        }, indent=2))
        print(f'Completed {sid}', flush=True)


if __name__ == '__main__':
    main()
