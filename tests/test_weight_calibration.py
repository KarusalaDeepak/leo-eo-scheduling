"""Protect the development/evaluation split used for weight selection."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_weight_calibration as calibration


class Calibration(unittest.TestCase):
    def test_heldout_scores_cannot_select_weights(self):
        frame=pd.DataFrame([
            dict(candidate=c,satellites=s,reward=float(c==1))
            for c in range(3) for s in [5,10,40,100]
        ])
        chosen=calibration.choose_candidate(frame,100)
        frame.loc[frame.satellites==100,'reward']=[1000,-1000,1000]
        self.assertEqual(calibration.choose_candidate(frame,100),chosen)
        self.assertEqual(chosen,1)

    def test_exact_ties_prefer_default(self):
        frame=pd.DataFrame([dict(candidate=c,satellites=s,reward=.5)
                            for c in range(3) for s in [5,100]])
        self.assertEqual(calibration.choose_candidate(frame,100),0)

    def test_rejects_missing_development_condition(self):
        frame=pd.DataFrame([dict(candidate=c,satellites=s,stress=t,reward=.5)
                            for c in range(2) for s in [5,100] for t in ['base','burst']])
        frame=frame.drop(frame[(frame.candidate==1)&(frame.satellites==5)&
                              (frame.stress=='burst')].index)
        with self.assertRaisesRegex(ValueError,'identical development'):
            calibration.choose_candidate(frame,100)

    def test_requires_development_group(self):
        frame=pd.DataFrame([dict(candidate=0,satellites=100,reward=.5)])
        with self.assertRaisesRegex(ValueError,'separate development'):
            calibration.choose_candidate(frame,100)

    def test_nonfinite_development_rejected_but_test_rewards_ignored(self):
        frame=pd.DataFrame([dict(candidate=c,satellites=s,reward=float(c))
                            for c in range(2) for s in [5,100]])
        frame.loc[frame.satellites==100,'reward']=np.nan
        self.assertEqual(calibration.choose_candidate(frame,100),1)
        frame.loc[frame.satellites==5,'reward']=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            calibration.choose_candidate(frame,100)

    def test_search_is_reproducible_and_scale_normalized(self):
        a=calibration.candidate_weights();b=calibration.candidate_weights()
        self.assertEqual(a,b);self.assertEqual(len(a),33)
        for weights in a:
            self.assertAlmostEqual(sum(weights.values()),1.)
            self.assertTrue(all(np.isfinite(v) and v>0 for v in weights.values()))


if __name__=='__main__':unittest.main()
