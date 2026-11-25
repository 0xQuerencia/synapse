#!/usr/bin/env python3
"""Offline artifact, semantic accessibility and conservative-claim checks."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parents[1]


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


def main():
    source = (SITE / 'dist/index.html').read_text()
    page = Page(source)
    errors = violations(page)
    assert not errors, errors
    internal = 0
    for node in page.nodes:
        attrs = node['attrs']
        for attr in ('href', 'src'):
            value = attrs.get(attr)
            if not value or value.startswith('#') or urlsplit(value).scheme:
                continue
            target = (SITE / 'dist' / unquote(urlsplit(value).path)).resolve()
            assert target.is_relative_to((SITE / 'dist').resolve()) and target.is_file(), f'missing asset/link: {value}'
            internal += 1
    css = (SITE / 'dist/assets/styles.css').read_text()
    assert 'prefers-reduced-motion:reduce' in css and 'scroll-behavior:auto' in css and 'animation:none!important' in css
    assert ':focus-visible' in css and 'outline:3px solid var(--accent)' in css
    for width in (1100, 800, 520):
        assert f'max-width:{width}px' in css
    for foreground, background in [('#edeef2', '#0d0f14'), ('#a5a8b6', '#0d0f14'), ('#b9adff', '#0d0f14'), ('#171322', '#b9adff')]:
        light, dark = sorted([luminance(foreground), luminance(background)], reverse=True)
        assert (light + .05) / (dark + .05) >= 4.5
    for phrase in ['Illustrative simulation', 'Fictional profiles.', 'No connected devices', 'Physical LAN/WAN validation is not attested', 'noncash usage']:
        assert phrase in source
    assert 'user-scalable=no' not in source
    assert '<form' in source and '<main' in source
    for unsafe in ['https://fonts.googleapis.com', 'google-analytics', 'gtag(', 'fake-live']:
        assert unsafe not in source
    # These failures correspond to common real regressions, rather than generated snapshots.
    assert 'document language missing' in violations(Page(source.replace('lang="en"', '')))
    assert 'duplicate IDs' in violations(Page(source.replace('<body>', '<body><div id="main"></div>')))
    assert 'image missing text alternative' in violations(Page(source.replace('<body>', '<body><img src="missing.svg">')))
    print(json.dumps({'semantic_checks': 'passed', 'internal_assets_and_links': internal, 'palette_contrast': 'four core pairs >= 4.5:1', 'reduced_motion': 'CSS policy checked; OS preference emulation not performed', 'responsive_rules': [1100, 800, 520], 'negative_regressions_rejected': 3}, indent=2))


if __name__ == '__main__':
    main()
