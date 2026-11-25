"""Optional local browser test for the synthetic dashboard, not a production user login."""
import argparse
import json
import tempfile
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from synapse_testnet.demo import setup
from synapse_testnet.http_api import Server
from synapse_testnet.synthetic import receipt_body,observed
from synapse_testnet.ledger import canonical

def run(out=None):
    with tempfile.TemporaryDirectory() as d:
        s,tokens,keys=setup(Path(d)/'db')
        with Server(s) as server:
            thread=threading.Thread(target=server.serve_forever); thread.start()
            try:
                with sync_playwright() as pw:
                    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
                    page=browser.new_page(viewport={'width':1280,'height':900}); errors=[]; page.on('pageerror',lambda err:errors.append(str(err)))
                    page.goto(f'http://127.0.0.1:{server.server_port}/'); page.get_by_label('Role access token').fill(tokens['client']); page.get_by_role('button',name='Connect',exact=True).click()
                    expect(page.locator('#available')).to_have_text('100'); page.get_by_role('button',name='Get quote').click(); expect(page.locator('#quote-result')).to_contain_text('20 test SC'); page.get_by_role('button',name='Reserve quoted test credits').click(); expect(page.locator('#held')).to_have_text('20')
                    for w,key in keys.items():
                        a=s.assignments(tokens[w])[0]; body=receipt_body(a,4); sig=key.sign(b'synapse-testnet-receipt-v1\0'+canonical(body)).hex(); s.receipt(tokens[w],body,sig)
                    for row in s.pending(tokens['verifier']):
                        body=json.loads(row['body']); reference=observed({**body,'first_layer':row['first_layer'],'end_layer':row['end_layer']},body['output_units']); s.validate(tokens['verifier'],row['assignment'],row['digest'],True,body['output_units'],reference)
                    page.get_by_role('button',name='Settle verified work').click(); expect(page.locator('#available')).to_have_text('88'); expect(page.locator('#held')).to_have_text('0'); expect(page.locator('#job')).to_contain_text('settled')
                    checks=[]
                    for width in (1280,768,390):
                        page.set_viewport_size({'width':width,'height':900}); assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'); checks.append({'viewport_width':width,'no_horizontal_overflow':True})
                    assert not errors,errors
                    if out: page.screenshot(path=str(out),full_page=True)
                    page.get_by_role('button',name='Disconnect').click(); expect(page.locator('#workspace')).to_be_hidden(); assert page.locator('#token').input_value()==''
                    browser.close(); return {'passed':True,'synthetic_only':True,'browser':'local headless Chromium','balance_after':88,'held_after':0,'viewport_checks':checks,'script_errors':len(errors)}
            finally: server.shutdown(); thread.join()
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--screenshot',type=Path); args=p.parse_args(); print(json.dumps(run(args.screenshot),indent=2))
