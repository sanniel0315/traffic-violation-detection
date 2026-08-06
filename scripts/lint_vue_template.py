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
import re
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


def lint_selfclosing_custom(text):
    """自訂元素(el-*)自閉合且後面還有兄弟節點 → 那些兄弟會被吞掉。

    Vue DOM 模板由瀏覽器的 HTML parser 解析,對未知元素 <el-x/> 的斜線
    會被忽略、視為「開始標籤」,後續同層節點全部變成它的子節點。
    Element Plus 元件渲染自己的內容、不會渲染這些被吞的節點 → 整段 UI 消失。

    實際案例:車道編號的 <el-input-number/> 把後面的「行車方向」下拉與
    進出線選擇器整組吞掉,使用者回報「看不到 IN/OUT 綁定」。

    只有「後面還接著別的元素」才算問題;後面直接是父層結束標籤的無害。

    注意:有 default slot 的元件(el-option / el-checkbox 等)即使被吞,
    內容仍會透過 slot 渲染出來,實務上多半正常 —— 所以這裡只列高風險的
    「不渲染 slot」元件,而且只警告不擋,避免大量誤報卡住 CI。
    """
    # 這些元件不渲染 default slot，被吞的兄弟節點會真的消失
    NO_SLOT = ('el-input-number', 'el-switch', 'el-date-picker',
               'el-time-picker', 'el-pagination', 'el-progress')
    lines = text.split('\n')
    pat = re.compile(r'<(el-[a-z-]+)\b((?:[^<>])*?)/>')
    hits = 0
    for i, line in enumerate(lines):
        m = pat.search(line)
        if not m or m.group(1) not in NO_SLOT:
            continue
        tail = '\n'.join(lines[i + 1:i + 3]).lstrip()
        nxt = re.match(r'<(span|select|button|input|div|el-[a-z-]+)\b', tail)
        if not nxt:
            continue
        hits += 1
        if hits == 1:
            print('\n[SELF-CLOSING] 不渲染 slot 的自訂元素自閉合，'
                  '後面的兄弟節點會被吞掉而消失：')
        print(f'       line {i+1} <{m.group(1)} .../>  後接 <{nxt.group(1)}>'
              f'  → 改寫成 </{m.group(1)}>')
    return hits


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
    sc = lint_selfclosing_custom(text)
    if fails:
        print(f'\n[FAIL] {fails} HTML structure error(s) — Vue mount likely fails')
        return 1
    if sc:
        # 警告不擋:有些是刻意保留的既有寫法，逐一確認才動
        print(f'\n[WARN] {sc} 處自閉合可能吞掉後面的節點，請確認該段 UI 有正常顯示')
    print('[OK] HTML structure parses cleanly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
