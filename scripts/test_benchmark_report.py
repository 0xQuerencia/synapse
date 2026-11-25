import math
import unittest
from benchmark_report import describe, percentile, timings
from local_cluster import verify_generation, verify_route


def events():
    return [
        {"event":"request_started", "elapsed_us":5000, "data":{}},
        {"event":"token", "elapsed_us":15000, "data":{"position":0,"token_id":1}},
        {"event":"token", "elapsed_us":17000, "data":{"position":1,"token_id":2}},
        {"event":"token", "elapsed_us":25000, "data":{"position":2,"token_id":3}},
        {"event":"generation_done", "elapsed_us":26000, "data":{"tokens_generated":3,"reason":"length"}},
    ]


class Benchmarks(unittest.TestCase):
    def test_real_percentiles_are_not_mean_or_target_count(self):
        values = [1.] * 19 + [100.]
        self.assertEqual(percentile(values,.95),1.)
        self.assertEqual(percentile(values,1),100.)
        self.assertNotEqual(describe(values)["mean_ms"],describe(values)["p95_ms"])

    def test_startup_is_excluded_and_actual_deltas_are_used(self):
        result = timings(events())
        self.assertEqual(result,{"actual_tokens":3,"ttft_ms":10.,"generation_ms":21.,"inter_token_ms":[2.,8.]})

    def test_unknown_samples_stay_missing(self):
        self.assertIsNone(describe([])["p95_ms"])
        for values in [[math.nan],[math.inf],[-1.]]:
            with self.assertRaises(ValueError): percentile(values,.95)

    def test_incomplete_or_duplicated_output_fails(self):
        for mutate in [lambda e:e.pop(),lambda e:e[2]["data"].update(position=0),lambda e:e[-1]["data"].update(tokens_generated=50)]:
            data=events();mutate(data)
            with self.assertRaises(RuntimeError): verify_generation(data)

    def test_live_route_must_cover_every_layer(self):
        route={"master_layer_range":{"start":0,"end":1},"worker_hops":[{"peer_id":"p","layer_range":{"start":1,"end":2}}]}
        record=[{"event":"status","data":{"route_plan":route}}]
        self.assertEqual(verify_route(record,2,1,False),route)
        route["worker_hops"][0]["layer_range"]["start"]=0
        with self.assertRaises(RuntimeError): verify_route(record,2,1,False)


if __name__ == "__main__": unittest.main()
