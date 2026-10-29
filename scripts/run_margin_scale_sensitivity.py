"""Exploratory scale sensitivity; not a search for a new deployment setting."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
import matched_scheduler as m

STRESSES = ['base', 'tight_deadline', 'reduced_bandwidth', 'high_volume', 'combined']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/margin_scale_sensitivity'))
    args = parser.parse_args()
    paths = sorted(args.dataset_dir.glob('*.json'))
    if not paths:
        raise ValueError('No dataset JSON files found.')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # Factor-of-two perturbations bracket the existing setting; they are not
    # estimated physical constants or independent validation data.
    scales = list(itertools.product([900., 1800., 3600.], [1., 2., 4.]))
    protocol = dict(scales=scales, default=[1800., 2.], weights=m.DEFAULT_WEIGHTS,
                    stresses=STRESSES, interpretation='Exploratory within-benchmark sensitivity; no tuning or default replacement',
                    dataset_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(m.__file__)]})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    for seconds, gb in scales:
        for path in paths:
            for stress in STRESSES:
                instance = m.load_instance(path, stress)
                instance['margin_scales'] = (seconds, gb)
                # evaluate independently validates every completed schedule.
                row, _ = m.evaluate(instance)
                rows.append(dict(row, time_scale_s=seconds, capacity_scale_gb=gb))
        print(f'Completed scales {seconds:g} s / {gb:g} GB', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'instance_results.csv', index=False)
    baseline = frame.query('time_scale_s == 1800 and capacity_scale_gb == 2')
    paired = frame.merge(baseline[['instance', 'stress', 'completion', 'reward']], on=['instance', 'stress'], suffixes=('', '_default'), validate='many_to_one')
    for metric in ['completion', 'reward']:
        paired[metric+'_delta_pp'] = 100*(paired[metric]-paired[metric+'_default'])
    paired.to_csv(out/'paired_deltas.csv', index=False)
    metrics = ['completion', 'reward', 'congestion', 'completion_delta_pp', 'reward_delta_pp']
    paired.groupby(['time_scale_s', 'capacity_scale_gb', 'stress'])[metrics].mean().to_csv(out/'condition_summary.csv')
    paired.groupby(['time_scale_s', 'capacity_scale_gb'])[metrics].mean().to_csv(out/'overall_summary.csv')


if __name__ == '__main__':
    main()
