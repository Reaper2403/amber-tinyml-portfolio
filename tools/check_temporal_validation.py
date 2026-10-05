#!/usr/bin/env python3
"""Exercise preserved temporal-evaluation helpers using synthetic trajectories."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('temporal_validation', ROOT / 'src/research/temporal_validation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    rows = []
    for user, n in [('synthetic_a', 20), ('synthetic_b', 41), ('synthetic_c', 100)]:
        for i in range(n):
            rows.append(dict(institute_year='synthetic', user_id=user, target_name='signal',
                             date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)))
    frame = module.assign_positions(pd.DataFrame(rows).sample(frac=1, random_state=42))
    for regime in ['expanding_refit', 'disjoint_refit']:
        for _, group in frame.groupby('user_id'):
            masks = {k: module.block_mask(group, regime, k) for k in
                     ['early_train', 'early_test', 'late_train', 'late_test']}
            for half in ['early', 'late']:
                train, test = masks[half + '_train'], masks[half + '_test']
                assert not (train & test).any()
                assert group.loc[train, 'date'].max() < group.loc[test, 'date'].min()
            assert not (masks['early_test'] & masks['late_test']).any()
            if regime == 'disjoint_refit':
                assert not (masks['early_train'] & masks['late_train']).any()
            else:
                assert (masks['early_train'] <= masks['late_train']).all()

    train = pd.DataFrame({'x': [0., 1., np.nan, 3.], 'constant': [4., 4., 4., 4.]})
    model = module.RidgeRegressor().fit(train.copy(), pd.Series([0., 1., 2., 3.]))
    state = [model.medians.copy(), model.means.copy(), model.scales.copy(), model.coef.copy()]
    prediction = model.predict(pd.DataFrame({'x': [1e6, np.nan], 'constant': [4., 4.]}))
    assert np.isfinite(prediction).all()
    for before, after in zip(state, [model.medians, model.means, model.scales, model.coef]):
        np.testing.assert_array_equal(before, after)
    print(json.dumps({'status': 'passed', 'synthetic_trajectories': 3, 'split_regimes': 2,
                      'checks': ['chronological order', 'train/test separation',
                                 'disjoint training blocks', 'expanding training inclusion',
                                 'prediction preserves train-only preprocessing'],
                      'scope': 'Helper checks on synthetic data; not reproduction of study results.'}, indent=2))


if __name__ == '__main__':
    main()
