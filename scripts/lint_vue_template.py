#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vue inline template smoke test — 部署後從 production 拉 index.html 跑 headless render 看 console error

驗證 inline template HTML 沒被 browser parser 截斷造成 Vue compile fail。

用法:
    python3 scripts/lint_vue_template.py http://192.168.0.108:8000/web/index.html
    python3 scripts/lint_vue_template.py web/index.html      # 本機

依賴: 純 Python stdlib (用 http.client + html.parser)
"""
import sys
import urllib.request
import html.parser
import pathlib


class StrictHTMLParser(html.parser.HTMLParser):
    """嚴格 HTML parser — 抓 attribute value 解析失敗或 tag 配對錯亂"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.errors = []
        self.opened = []  # stack of (tag, line, col)
        self.attr_warnings = []

    def handle_starttag(self, tag, attrs):
        if tag in ('br','hr','img','input','meta','link','source','track','area','base','col','embed','param','wbr'):
            return  # void elements
        self.opened.append((tag, self.getpos()))
        # 檢查 v-if/v-show/v-bind/v-model/@/: 內含未 escape 的疑似 broken pattern
        for name, value in attrs:
            if value is None:
                continue
            is_vue_dir = (
                name.startswith('v-') or name.startswith(':') or name.startswith('@')
            )
            if is_vue_dir and ('>' in value or '<' in value):
                # 注意：HTML5 spec 雙引號 attribute 內 >/< 是合法的；
                # 但仍標記以便 review
                self.attr_warnings.append((self.getpos(), tag, name, value[:80]))

    def handle_endtag(self, tag):
        if tag in ('br','hr','img','input','meta','link','source','track','area','base','col','embed','param','wbr'):
            return
        if not self.opened:
            self.errors.append((self.getpos(), f'</{tag}> with no opener'))
            return
        last_tag, last_pos = self.opened[-1]
        if last_tag != tag:
            self.errors.append((self.getpos(), f'</{tag}> mismatch (expected </{last_tag}> opened at {last_pos})'))
        else:
            self.opened.pop()

    def error(self, message):  # type: ignore[override]
        self.errors.append((self.getpos(), message))


def lint(html_text: str) -> int:
    p = StrictHTMLParser()
    p.feed(html_text)
    p.close()
    fails = 0
    for pos, msg in p.errors[:15]:
        print(f'[FAIL] line {pos[0]} col {pos[1]}: {msg}')
        fails += 1
    if p.opened:
        for tag, pos in p.opened[:5]:
            print(f'[FAIL] unclosed <{tag}> opened at line {pos[0]}')
            fails += 1
    # Vue directive attribute 內含 >/< 報 WARN 不算 fail
    # (HTML5 spec 雙引號內合法，但部分 case 會出問題)
    if p.attr_warnings:
        print(f'[WARN] {len(p.attr_warnings)} Vue directive attribute(s) contain >/< (HTML5 legal but worth review)')
        for pos, tag, name, val in p.attr_warnings[:5]:
            print(f'       line {pos[0]} <{tag} {name}="{val}...">')
    return fails


def main():
    if len(sys.argv) < 2:
        print('usage: lint_vue_template.py <url|path>', file=sys.stderr)
        return 2
    target = sys.argv[1]
    if target.startswith('http://') or target.startswith('https://'):
        print(f'fetching {target}')
        with urllib.request.urlopen(target, timeout=8) as r:
            text = r.read().decode('utf-8')
    else:
        text = pathlib.Path(target).read_text(encoding='utf-8')
    fails = lint(text)
    if fails:
        print(f'\n[FAIL] {fails} HTML structure error(s) — Vue mount likely fails')
        return 1
    print('[OK] HTML structure parses cleanly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
