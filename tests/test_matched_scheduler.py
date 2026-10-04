"""Regression tests for the feasibility defects found during manuscript review."""
import sys
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import matched_scheduler as m


def fixture():
    task=dict(id='t',arrival=0,deadline=100,priority=5,
              atoms=[dict(id='a',duration=10,volume=1)])
    instance=dict(tasks=[task],obs={'a':[('A',0,10)]},
                  downlinks={'B':[('G',30,50,1.)]},
                  isl={'A':[('B',10,30,1.)]},storage={'A':2.,'B':2.})
    return instance,task


class Constraints(unittest.TestCase):
    def test_short_observation_not_extended(self):
        i,t=fixture();i['obs']['a']=[('A',0,9)]
        self.assertIsNone(m.build_plan(i,t,m.Ledger(),('earliest','early')))

    def test_relay_both_endpoints_reserved(self):
        i,t=fixture();p=m.build_plan(i,t,m.Ledger(),('earliest','early'))
        self.assertIsNotNone(p)
        self.assertIn(('sat:A',10,18),p.reservations)
        self.assertIn(('sat:B',10,18),p.reservations)
        m.validate_schedule(i,[p])

    def test_relay_conflict_and_bandwidth(self):
        i,t=fixture();busy=m.Ledger(busy={'sat:B':[(10,30)]})
        self.assertIsNone(m.build_plan(i,t,busy,('earliest','early')))
        i['isl']['A']=[('B',10,17,1.)]
        self.assertIsNone(m.build_plan(i,t,m.Ledger(),('earliest','early')))

    def test_ground_station_conflict(self):
        i,t=fixture();busy=m.Ledger(busy={'gs:G':[(30,50)]})
        self.assertIsNone(m.build_plan(i,t,busy,('earliest','early')))

    def test_storage_limit(self):
        i,t=fixture();i['storage']['B']=.5
        self.assertIsNone(m.build_plan(i,t,m.Ledger(),('earliest','early')))

    def test_failed_trial_does_not_reserve(self):
        i,t=fixture();i['downlinks']={};ledger=m.Ledger()
        self.assertIsNone(m.build_plan(i,t,ledger,('earliest','early')))
        self.assertEqual(ledger.busy,{})
        self.assertEqual(ledger.storage,{})

    def test_segments_cannot_overlap(self):
        i,t=fixture();i['downlinks']={'A':[('G',20,30,.4),('H',25,35,.4)]}
        i['isl']={}
        self.assertIsNone(m.build_plan(i,t,m.Ledger(),('earliest','early')))

    def test_split_downlink_conserves_payload(self):
        i,t=fixture();i['downlinks']={'A':[('G',20,30,.4),('G',30,40,.4)]}
        p=m.build_plan(i,t,m.Ledger(),('earliest','early'))
        self.assertEqual(len(p.deliveries),2)
        self.assertAlmostEqual(sum(d[5] for d in p.deliveries),1.)
        m.validate_schedule(i,[p])

    def test_validator_catches_missing_relay_reservation(self):
        i,t=fixture();p=m.build_plan(i,t,m.Ledger(),('earliest','early'))
        p.reservations.remove(('sat:B',10,18))
        with self.assertRaises(AssertionError):m.validate_schedule(i,[p])

    def test_validator_catches_false_payload(self):
        i,t=fixture();p=m.build_plan(i,t,m.Ledger(),('earliest','early'))
        p.volume=2.
        with self.assertRaises(AssertionError):m.validate_schedule(i,[p])

    def test_score_actually_selects_alternative(self):
        i,t=fixture();p=m.build_plan(i,t,m.Ledger(),('earliest','early'))
        alternative=replace(p,margin=1.,congestion=.01,risk=0.,variant='alternative')
        self.assertIs(m.select([p,alternative],'racs',np.random.default_rng(0)),alternative)
        self.assertEqual(m.select([p,alternative],'margin',np.random.default_rng(0)).variant,'alternative')

    def test_no_forty_satellite_candidate_cap(self):
        i,t=fixture()
        # Only the 41st satellite can observe: the first 40 are already occupied.
        sats=[f'S{k:02}' for k in range(1,42)]
        i['obs']['a']=[(sat,0,10) for sat in sats]
        i['storage']={sat:2. for sat in sats}
        i['isl']={}
        i['downlinks']={sats[-1]:[('G',20,30,1.)]}
        ledger=m.Ledger(busy={'sat:'+sat:[(0,10)] for sat in sats[:-1]})
        plans=m.candidate_plans(i,t,ledger)
        self.assertTrue(plans)
        self.assertTrue(all(p.observations[0][1]=='S41' for p in plans))
        for p in plans:m.validate_schedule(i,[p])

    def test_future_visibility_is_eligible_in_batch_plan(self):
        i,t=fixture();i['obs']['a']=[('A',50,60)]
        i['isl']={};i['downlinks']={'A':[('G',70,80,1.)]}
        plans=m.candidate_plans(i,t,m.Ledger())
        self.assertTrue(plans)
        self.assertEqual(plans[0].observations[0][2],50)
        m.validate_schedule(i,[plans[0]])

    def test_positive_weight_scaling_preserves_ranking(self):
        i,t=fixture();p=m.build_plan(i,t,m.Ledger(),('earliest','early'))
        q=replace(p,margin=1.,congestion=.01,risk=0.,variant='alternative')
        expected=m.select([p,q],'racs',np.random.default_rng(0))
        for factor in [.5,.75,1.,1.25,1.5]:
            weights={k:factor*v for k,v in m.DEFAULT_WEIGHTS.items()}
            self.assertIs(m.select([p,q],'racs',np.random.default_rng(0),weights),expected)


if __name__=='__main__': unittest.main()
