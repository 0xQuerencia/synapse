"""Source candidate gate summary. Never self-certifies production readiness."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--assert-release',action='store_true'); args=p.parse_args()
    root=Path(__file__).resolve().parents[2]; state=json.loads((root/'docs/community/gates.json').read_text()); gates=state['gates']; assert {g['id'] for g in gates}=={f'G{i}' for i in range(7)}
    # Hard stop for this candidate: owner and independent acceptance are absent.
    assert all(g['accepted'] is False for g in gates)
    print(json.dumps({'scope':'additive local source candidate','release_approved':False,'open_gates':[g['id'] for g in gates],'production_credit_settlement':False,'public_swarm':False},indent=2))
    return 1 if args.assert_release else 0
if __name__=='__main__':raise SystemExit(main())
