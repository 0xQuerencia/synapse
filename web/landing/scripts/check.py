#!/usr/bin/env python3
"""Offline artifact, semantic accessibility and conservative-claim checks."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

DIST = Path(__file__).resolve().parents[1] / 'dist'


class Page(HTMLParser):
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, source):
        super().__init__()
        self.nodes, self.stack = [], []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        node = {'tag': tag, 'attrs': dict(attrs), 'ancestors': self.stack.copy(), 'text': ''}
        self.nodes.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1]['tag'] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]['tag'] == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, text):
        for node in self.stack:
            node['text'] += text


def violations(page):
    errors = []
    ids = [n['attrs']['id'] for n in page.nodes if 'id' in n['attrs']]
    if len(ids) != len(set(ids)):
        errors.append('duplicate IDs')
    if not any(n['tag'] == 'html' and n['attrs'].get('lang') == 'en' for n in page.nodes):
        errors.append('document language missing')
    for tag in ['main', 'h1']:
        if sum(n['tag'] == tag for n in page.nodes) != 1:
            errors.append(f'exactly one {tag} required')
    labels = {n['attrs'].get('for'): n['text'].strip() for n in page.nodes if n['tag'] == 'label'}
    for node in page.nodes:
        a, tag = node['attrs'], node['tag']
        if tag == 'img' and 'alt' not in a:
            errors.append('image missing text alternative')
        if tag in {'input', 'select', 'textarea'} and a.get('type') != 'hidden':
            if not (a.get('aria-label') or a.get('aria-labelledby') or labels.get(a.get('id')) or any(p['tag'] == 'label' and p['text'].strip() for p in node['ancestors'])):
                errors.append('control missing accessible name')
        if tag == 'button' and not (node['text'].strip() or a.get('aria-label')):
            errors.append('button missing accessible name')
        if tag in {'a', 'button', 'input', 'select', 'summary'} and a.get('tabindex', '0').isdigit() and int(a.get('tabindex', '0')) > 0:
            errors.append('positive tabindex distorts keyboard order')
        if tag == 'a' and a.get('href', '').startswith('#') and len(a['href']) > 1 and a['href'][1:] not in ids:
            errors.append('broken anchor: ' + a['href'])
    return errors


def luminance(color):
    values = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return sum(weight * (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) for weight, x in zip([.2126, .7152, .0722], values))


def check_links(page):
    internal = 0
    for node in page.nodes:
        for attr in ('href', 'src'):
            value = node['attrs'].get(attr)
            if not value or value.startswith('#') or urlsplit(value).scheme:
                continue
            target = (DIST / unquote(urlsplit(value).path)).resolve()
            if target.is_dir():
                target /= 'index.html'
            assert target.is_relative_to(DIST.resolve()) and target.is_file(), f'missing asset/link: {value}'
            internal += 1
    return internal


def check_trace(page):
    controls = [n for n in page.nodes if 'data-trace-step' in n['attrs']]
    panels = [n for n in page.nodes if 'data-trace-panel' in n['attrs']]
    expected = ['route', 'prefill', 'decode', 'result']
    assert [n['attrs']['data-trace-step'] for n in controls] == expected
    assert [n['attrs']['data-trace-panel'] for n in panels] == expected
    assert [n['attrs']['aria-pressed'] for n in controls] == ['true', 'false', 'false', 'false']
    assert ['hidden' in n['attrs'] for n in panels] == [False, True, True, True]
    assert all(n['tag'] == 'button' and n['attrs'].get('type') == 'button' for n in controls)
    assert any(n['attrs'].get('class') == 'trace-fallback' and 'hidden' not in n['attrs'] and 'seven decode passes' in n['text'] for n in page.nodes)


def main():
    landing = (DIST / 'index.html').read_text()
    simulator = (DIST / 'simulator.html').read_text()
    internal = 0
    for source in [landing, simulator]:
        page = Page(source)
        errors = violations(page)
        assert not errors, errors
        internal += check_links(page)
        for forbidden in ['user-scalable=no', 'https://fonts.googleapis.com', 'google-analytics', 'gtag(']:
            assert forbidden not in source
    check_trace(Page(landing))
    for phrase in ['Recorded fixture run', 'Private Tailscale', 'Layer 0', 'Layer 1', '8 of 8', 'Still experimental']:
        assert phrase in landing
    for phrase in ['Illustrative simulation', 'Fictional profiles.', 'No connected devices', '<form']:
        assert phrase in simulator

    css = (DIST / 'assets/styles.css').read_text()
    assert '@media (prefers-reduced-motion: reduce)' in css and 'animation: none !important' in css
    assert ':focus-visible' in css and 'outline: 3px solid var(--accent)' in css
    colors = dict(re.findall(r'(--[a-z-]+): (#[0-9a-f]{6})', css))
    pairs = [('ink', 'paper'), ('muted', 'paper'), ('muted', 'paper-deep'), ('accent', 'paper'),
             ('trace-paper', 'trace-bg'), ('trace-muted', 'trace-bg'), ('trace-accent', 'trace-bg'),
             ('ink', 'trace-accent')]
    for foreground, background in pairs:
        light, dark = sorted([luminance(colors['--' + foreground]), luminance(colors['--' + background])], reverse=True)
        assert (light + .05) / (dark + .05) >= 4.5, (foreground, background)

    # Exercise common semantic regressions.
    assert 'document language missing' in violations(Page(landing.replace('lang="en"', '')))
    assert 'duplicate IDs' in violations(Page(landing.replace('<main', '<div id="main"></div><main')))
    assert 'image missing text alternative' in violations(Page(landing.replace('<main', '<img src="missing.svg"><main')))
    print(json.dumps({'semantic_checks': 'passed for both pages', 'internal_assets_and_links': internal,
                      'palette_contrast_pairs': len(pairs), 'trace_steps': 4, 'reduced_motion': 'CSS policy checked',
                      'negative_regressions_rejected': 3}, indent=2))


if __name__ == '__main__':
    main()
