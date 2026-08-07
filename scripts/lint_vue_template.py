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


# 不渲染 default slot 的元件:被吞掉的兄弟節點會真的消失。
# el-input 是 2026-08-07 用瀏覽器實測補上的 —— roi_editor.html 的
# <el-input v-model="item.zone.name" .../> 把整列的下拉、點數、
# 甚至「儲存」「刪除」按鈕全吞掉,而舊清單沒有它,跑 lint 也抓不到。
NO_SLOT = (
    'el-input', 'el-input-number', 'el-switch', 'el-date-picker',
    'el-time-picker', 'el-pagination', 'el-progress', 'el-slider',
    'el-rate', 'el-color-picker', 'el-image', 'el-avatar',
)


def _scan_selfclosing(text):
    """逐字掃出所有自閉合的自訂元素,回傳 [(tag, 位置, 標籤後方的內容)]。

    不用 regex:惰性量詞會跨過標籤邊界配對(實際踩過:把 <el-option/>
    關成 </el-select>,整份模板被改壞)。掃描器逐字讀,遇到 <el- 讀標籤名,
    再往後掃到「該開始標籤自己的 >」(途中跳過引號內容),不可能跨標籤。
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith('<el-', i):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == '-'):
                j += 1
            tag = text[i + 1:j]
            k, quote = j, None
            while k < n:
                c = text[k]
                if quote:
                    if c == quote:
                        quote = None
                elif c in '"\'':
                    quote = c
                elif c == '>':
                    break
                k += 1
            if k < n and text[i:k].rstrip().endswith('/'):
                out.append((tag, i, text[k + 1:k + 400]))
            i = k + 1 if k < n else n
            continue
        i += 1
    return out


def lint_selfclosing_custom(text):
    """自訂元素(el-*)自閉合且後面還有兄弟節點 → 那些兄弟會被吞掉。

    Vue DOM 模板由瀏覽器的 HTML parser 解析,對未知元素 <el-x/> 的斜線
    會被忽略、視為「開始標籤」,後續同層節點全部變成它的子節點。
    不渲染 default slot 的元件不會把它們畫出來 → 整段 UI 消失。

    實際案例:
    - 車道編號的 <el-input-number/> 吞掉「行車方向」下拉與進出線選擇器
    - roi_editor.html 的 <el-input/> 吞掉整列,連儲存/刪除按鈕都不見

    只有「後面還接著別的開始標籤」才算問題;後面直接是父層結束標籤的無害。
    有 default slot 的元件(el-option / el-checkbox / el-button 等)即使被吞,
    內容仍會透過 slot 渲染,實務上多半正常 —— 只警告不擋,避免誤報卡住 CI。
    """
    hits = 0
    for tag, pos, tail in _scan_selfclosing(text):
        if tag not in NO_SLOT:
            continue
        # 跳過空白與註解,看下一個實質內容是不是「開始標籤」
        t = tail.lstrip()
        while t.startswith('<!--'):
            end = t.find('-->')
            if end < 0:
                break
            t = t[end + 3:].lstrip()
        m = re.match(r'<(?!/)([a-z][a-z0-9-]*)', t)
        if not m:
            continue                      # 後面是 </parent> 或文字 → 無害
        line = text.count('\n', 0, pos) + 1
        hits += 1
        if hits == 1:
            print('\n[SELF-CLOSING] 不渲染 slot 的自訂元素自閉合，'
                  '後面的兄弟節點會被吞掉而消失：')
        if hits <= 20:
            print(f'       line {line} <{tag} .../>  後接 <{m.group(1)}>'
                  f'  → 改寫成 </{tag}>')
    if hits > 20:
        print(f'       ...(還有 {hits - 20} 處)')
    return hits


def discover_inline_template_pages(root='web'):
    """找出所有「inline template」頁面 —— 有 createApp 且模板寫在 HTML 裡。

    這類頁面才會踩到瀏覽器 HTML parser 的坑。以前 SOP 只要求檢查
    index.html,結果 roi_editor.html 的自閉合 bug 藏了很久沒人發現
    (整列的儲存/刪除按鈕都是隱形的)。
    """
    base = pathlib.Path(root)
    if not base.is_dir():
        return []
    found = []
    for f in sorted(base.glob('*.html')):
        try:
            s = f.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if 'createApp' in s and '<div id="app"' in s:
            found.append(f)
    return found


def lint_one(target):
    """回傳 (fails, warns)。"""
    if str(target).startswith(('http://', 'https://')):
        print(f'fetching {target}')
        with urllib.request.urlopen(str(target), timeout=8) as r:
            text = r.read().decode('utf-8')
    else:
        text = pathlib.Path(target).read_text(encoding='utf-8')
    fails = lint(text)
    sc = lint_selfclosing_custom(text)
    if fails:
        print(f'[FAIL] {fails} HTML structure error(s) — Vue mount likely fails')
    elif sc:
        print(f'[WARN] {sc} 處自閉合可能吞掉後面的節點，請確認該段 UI 有正常顯示')
    else:
        print('[OK] HTML structure parses cleanly')
    return fails, sc


def main():
    args = sys.argv[1:]
    if args and args[0] == '--all':
        args = []
    targets = args or discover_inline_template_pages()
    if not targets:
        print('usage: lint_vue_template.py <url|path> [more paths...]', file=sys.stderr)
        print('       lint_vue_template.py --all    # 掃 web/ 下所有 inline-template 頁',
              file=sys.stderr)
        return 2

    total_fail = total_warn = 0
    results = []
    for t in targets:
        print(f'\n===== {t} =====')
        f, w = lint_one(t)
        total_fail += f
        total_warn += w
        results.append((str(t), f, w))

    if len(results) > 1:
        print('\n===== 總結 =====')
        for name, f, w in results:
            status = 'FAIL' if f else ('WARN' if w else 'OK')
            print(f'  {status:<5} {name}   結構錯誤 {f}  自閉合風險 {w}')
        print(f'  合計: 結構錯誤 {total_fail}, 自閉合風險 {total_warn}')
    return 1 if total_fail else 0


if __name__ == '__main__':
    sys.exit(main())
