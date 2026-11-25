import unittest
from synapse_testnet.economy import SCENARIOS,simulate
class EconomyTests(unittest.TestCase):
    def test_all_scenarios_conserve(self):
        for s in SCENARIOS:
            for seed in (1,42,100):
                result=simulate(s,seed); self.assertTrue(result['conserved']); self.assertEqual(result['issued'],result['treasury']+result['requester_credits']+result['provider_credits']+result['escrow']); self.assertEqual(result['escrow'],0)
    def test_determinism(self): self.assertEqual(simulate(SCENARIOS[0]),simulate(SCENARIOS[0]))
    def test_zero_demand(self): self.assertTrue(all(n==0 for n in simulate(SCENARIOS[1])['outcomes'].values()))
    def test_zero_supply(self): self.assertEqual(simulate(SCENARIOS[2])['outcomes'].get('completed',0),0)
    def test_grant_exhaustion(self): self.assertGreater(simulate(SCENARIOS[-1])['outcomes']['grant_shortfall'],0)
