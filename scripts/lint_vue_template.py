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


# Vue 3 global build 的 composition API —— 這些一定要從 Vue 解構出來才能用
_VUE_APIS = (
    'ref', 'reactive', 'computed', 'watch', 'watchEffect', 'nextTick',
    'onMounted', 'onUnmounted', 'onBeforeMount', 'onBeforeUnmount',
    'onUpdated', 'onActivated', 'onDeactivated', 'defineComponent',
    'shallowRef', 'toRaw', 'markRaw', 'provide', 'inject',
)
_VUE_DESTRUCTURE = re.compile(r'const\s*\{([^}]*)\}\s*=\s*Vue\b')


def lint_vue_api_imports(text):
    """抓「用了 Vue API 但沒從 Vue 解構出來」。

    🛑 這種錯誤語法完全合法,HTML 結構檢查與 JS 語法檢查都抓不到 ——
       要等執行到那一行才 ReferenceError,而 Vue 的 setup 一拋例外整個 app
       就不掛載 → 白畫面。實際踩過兩次:
       - 2026-08-15 roi_editor.html 用 onUnmounted 但沒引入 → 編輯器空白頁
       - 同類型:識別字在模板用到但 setup 沒回傳 → 主頁白畫面
    """
    imported = set()
    for m in _VUE_DESTRUCTURE.finditer(text):
        for name in m.group(1).split(','):
            name = name.split(':')[0].strip()
            if name:
                imported.add(name)
    if not imported:
        return []          # 沒有用解構寫法(可能是 Vue.ref 形式),不做判斷
    missing = []
    for api in _VUE_APIS:
        if api in imported:
            continue
        # 呼叫形式 api(...) 且前面不是 . 或字母(排除 Vue.ref / myRef 之類)
        m = re.search(r'(?<![.\w])' + api + r'\s*\(', text)
        if m:
            line = text[:m.start()].count('\n') + 1
            missing.append((api, line))
    return missing


_SETUP_RETURN = re.compile(r'\n\s*return\s*\{([^}]*)\}\s*;?\s*\n\s*\}\s*\n?\s*\}\)', re.S)
# 模板裡的函式呼叫:{{ foo(...) }} 或 :attr="foo(...)" / @evt="foo(...)"
# 排除 $ 前綴:$t / $lang 這類是 app.config.globalProperties 提供的,
# 不會出現在 setup return 裡(排除掉才不會誤報)。
_TPL_CALL = re.compile(r'[:@][\w.-]+\s*=\s*"[^"]*?(?<![.\w$])([a-zA-Z_]\w*)\s*\(')
_JS_BUILTINS = {
    'String', 'Number', 'Boolean', 'Array', 'Object', 'Math', 'Date', 'JSON',
    'parseInt', 'parseFloat', 'isNaN', 'encodeURIComponent', 'decodeURIComponent',
    'if', 'for', 'while', 'return', 'typeof', 'new', 'function', 'catch',
}


def lint_setup_return(text):
    """抓「模板呼叫了某個函式,但 setup 的 return 沒有它」。

    🛑 這種錯誤只有在那段模板「實際被渲染」時才爆 —— v-for / v-if 包住的
       區塊平常不算,一旦有資料就 render error → 整個 app unmount → 白畫面。
       實際踩過兩次:
       - roi_editor.html zoneColorOf:zone 列表是 v-for,存檔後列表出現才白
       - index.html deviceIdentity:硬體監測頁點開才白
       抓不到 setup return 區塊時直接跳過(不同寫法很多),寧可漏報不要誤報。
    """
    # 🛑 不可以用「第一個符合的 return {...}」—— 巢狀函式裡的小 return 也會中,
    #    抓到它就等於拿到一份幾乎空的清單,模板裡每個函式都變成「沒回傳」
    #    (實測 index.html 一次噴 98 個假警報)。setup 的 return 是全檔最大的那個,
    #    取識別字最多的那一個才穩。
    best = None
    for m in _SETUP_RETURN.finditer(text):
        names = {n.split(':')[0].strip() for n in m.group(1).split(',') if n.strip()}
        if best is None or len(names) > len(best):
            best = names
    if not best or len(best) < 20:
        return []          # 找不到夠大的 setup return → 跳過,寧可漏報不要誤報
    returned = best
    # 🛑 不能用「第一個 <script 之前」當模板 —— 檔頭就有 <script src=...> 載入
    #    Vue/ElementPlus,那樣切出來的模板是空的,整個檢查等於沒跑(實際踩過)。
    #    正確做法:把所有 <script>...</script> 區塊挖掉,剩下的才是模板。
    tpl = re.sub(r'<script\b[^>]*>.*?</script>', ' ', text, flags=re.S | re.I)
    missing, seen = [], set()
    for cm in _TPL_CALL.finditer(tpl):
        name = cm.group(1)
        if name in returned or name in _JS_BUILTINS or name in seen:
            continue
        # 該名稱要真的在 script 裡被定義過,否則可能是內建/全域
        if not re.search(r'(?:const|let|var|function)\s+' + name + r'\b', text):
            continue
        seen.add(name)
        missing.append((name, tpl[:cm.start()].count('\n') + 1))
    return missing


# ── watch 的暫時死區(TDZ)檢查 ────────────────────────────────────────────
# 🛑 2026-09-05 踩到:新加的 watch(sigStrategyVal, ...) 放在 setup 中段,
#    而 sigStrategyVal 這個 computed 讀的 sigSafety 在兩百多行之後才宣告。
#    watch 會**立即求值** source 拿初始值 → 讀到還在 TDZ 的 const →
#    ReferenceError → 整個 setup 中斷 → 整頁白畫面(連登入都動不了)。
#    node --check 只看語法,div 平衡也正常,兩道既有關卡都看不見它。
#    computed 本身是惰性的沒問題,問題出在「watch 把它變成立即求值」。
_DECL_RE = re.compile(r'^\s*(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=', re.M)
_IDENT_RE = re.compile(r'\b([A-Za-z_$][\w$]*)\b')
_WATCH_RE = re.compile(r'\bwatch\s*\(')


def _first_arg(text, open_paren_idx):
    """取 watch( 的第一個引數原文。遇到深度 0 的逗號才算分隔。"""
    depth = 0
    i = open_paren_idx
    start = i + 1
    while i < len(text):
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                return text[start:i]
        elif c == ',' and depth == 1:
            return text[start:i]
        i += 1
    return ''


def _balanced(text, open_idx, limit=8000):
    """從 open_idx 的 ( 開始做括號配對,回傳整個呼叫的內容(不含最外層括號)。

    只用來界定「這一個 watch 呼叫」的範圍。抓不到配對(超過 limit)就回 None,
    寧可漏報也不要拿一段亂截的文字去比對 —— 亂截會製造大量誤報
    (2026-09-06 用固定長度截 1200 字元,一口氣噴出 42 個假陽性)。
    """
    depth = 0
    j = open_idx
    end = min(len(text), open_idx + limit)
    while j < end:
        c = text[j]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j]
        j += 1
    return None

def lint_watch_tdz(text):
    """回傳 [(watch 行號, 識別字, 宣告行號)]。"""
    # 直接掃全檔:const/let 宣告本來就只出現在 inline <script> 裡,
    # 而且用全檔 offset 算出來的行號可以直接對應到檔案,回報時不必再換算。
    script = text
    # 每個識別字第一次被 const/let 宣告的位置(字元 offset)
    decl = {}
    for m in _DECL_RE.finditer(script):
        # 只採計 setup 頂層的宣告。函式內的區域 const 不會造成 TDZ ——
        # 它跟 watch 不在同一個作用域。先前沒濾,roi_editor.html 的 watch 被誤報:
        # 它讀到 computed 本體裡的物件鍵 label,而 label 是幾百行後某個
        # 繪圖函式內的區域變數(2026-09-06 誤報)。
        # 🛑 _DECL_RE 的 ^\s* 會把行首縮排吃進 match,所以縮排要從 match 內容量,
        #    不能用 m.start() 減行首(那永遠是 0)。
        indent = len(m.group(0)) - len(m.group(0).lstrip())
        if indent > 10:   # setup 頂層是 8 空格;12 以上都是巢狀函式內
            continue
        decl.setdefault(m.group(1), m.start())
    # computed 的本體,供一層追進去用
    bodies = {}
    for name, off in decl.items():
        line_end = script.find('\n', off)
        head = script[off:line_end if line_end != -1 else len(script)]
        if 'computed(' in head:
            # 抓到該 computed 的結尾括號為止(夠用即可,不做完整 parse)
            oi = script.index('(', script.index('computed(', off) + 7)
            depth, j = 0, oi
            while j < len(script):
                if script[j] in '([{':
                    depth += 1
                elif script[j] in ')]}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            bodies[name] = script[oi:j]

    hits = []
    for wm in _WATCH_RE.finditer(script):
        oi = script.index('(', wm.start())
        arg = _first_arg(script, oi)
        # 直接用到的識別字,加上「若它是 computed 就再看它的本體」一層
        names = set(_IDENT_RE.findall(arg))
        # 🛑 immediate:true 的 watch,回呼在 setup 當下就跑一次 ——
        #    回呼裡讀到的識別字一樣會 TDZ。只看第一個參數會漏掉
        #    (2026-09-06:analytics_insights 的 watch 回呼讀 cameras,
        #     cameras 宣告在後面,深層連結進去整頁白畫面)。
        call = _balanced(script, oi)
        if call is not None and 'immediate' in call and 'true' in call:
            cand = set(_IDENT_RE.findall(call))
            # 排除物件字面值的鍵(watch 的 {immediate:true, flush:'post'} 這種),
            # 以及回呼自己的參數名 —— 兩者都不是外層作用域的識別字,算進去全是誤報。
            cand -= set(re.findall(r'([A-Za-z_$][\w$]*)\s*:', call))
            for pm in re.finditer(r'\(([^()]*)\)\s*=>', call):
                cand -= set(re.findall(r'[A-Za-z_$][\w$]*', pm.group(1)))
            names |= cand
        for n in list(names):
            if n in bodies:
                names |= set(_IDENT_RE.findall(bodies[n]))
        for n in sorted(names):
            off = decl.get(n)
            if off is not None and off > wm.start():
                hits.append((script[:wm.start()].count('\n') + 1, n,
                             script[:off].count('\n') + 1))
    return hits


SVG_CAMEL_ATTRS = (
    "viewBox", "preserveAspectRatio", "gradientUnits", "gradientTransform",
    "patternUnits", "patternContentUnits", "clipPathUnits", "maskUnits",
    "markerWidth", "markerHeight", "refX", "refY", "textLength", "lengthAdjust",
    "spreadMethod", "startOffset", "baseFrequency", "stdDeviation",
)


def _kebab(name):
    return re.sub(r'([A-Z])', lambda m: '-' + m.group(1).lower(), name)


def lint_svg_camel(text):
    """v-bind 到 SVG 的 camelCase 屬性必須加 .camel 修飾詞。

    inline template 由瀏覽器的 HTML parser 先解析,屬性名一律轉小寫。
    :viewBox="..." 到 Vue 手上已經是 :viewbox,設出來的是 viewbox 屬性 ——
    SVG 屬性名區分大小寫,瀏覽器直接忽略。後果不是報錯,是默默壞掉:
    沒有 viewBox 的 SVG 改用 1 使用者單位 = 1px,內容只佔容器一角。
    2026-09-06 使用者回報「控制時間軸只有 container 一半」就是這個。
    正確寫法::view-box.camel="..."
    """
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        for attr in SVG_CAMEL_ATTRS:
            if attr.lower() == attr:
                continue
            for form in (":" + attr + "=", "v-bind:" + attr + "="):
                if form in line:
                    bad.append((i, attr, line.strip()[:90]))
    return bad

def lint_reactive_dot_value(text):
    """reactive(...) 宣告的物件不可以用 .value 存取。

    ref 要 .value、reactive 不要,兩者混用不會有語法錯,只會在執行時丟
    TypeError: Cannot set properties of undefined —— 而且常常是在事件處理器裡,
    畫面不會整頁掛掉,只有那個功能默默失效。
    2026-09-06:sigLiveErr 是 reactive 卻寫成 sigLiveErr.value[id]=false,
    導致影像載入完成的旗標永遠清不掉,戰情的相機格被深色遮罩蓋住。
    """
    bad = []
    decls = set(re.findall(r'(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*reactive\s*\(', text))
    for name in sorted(decls):
        for m in re.finditer(r'' + re.escape(name) + r'\.value', text):
            ln = text[:m.start()].count(chr(10)) + 1
            bad.append((ln, name))
    return bad

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
    api_missing = lint_vue_api_imports(text)
    if api_missing:
        # 這個一定是 FAIL 不是 WARN —— 執行到就 ReferenceError,整頁白畫面
        print(f'[FAIL] {len(api_missing)} 個 Vue API 有用到但沒從 Vue 解構出來 '
              f'(執行時 ReferenceError → 整頁白畫面):')
        for api, line in api_missing:
            print(f'       第 {line} 行用到 {api}() —— 請加進 const {{ ... }} = Vue')
        fails += len(api_missing)
    rdv = lint_reactive_dot_value(text)
    for ln, name in rdv:
        print("[FAIL] %s:%s %s 是 reactive() 宣告的,不可以用 .value 存取"
              " —— 執行時會丟 TypeError,該功能默默失效" % (target, ln, name))
    fails += len(rdv)
    svgc = lint_svg_camel(text)
    for ln, attr, snippet in svgc:
        print("[FAIL] %s:%s SVG 屬性 :%s 在 inline template 會被轉小寫而失效,"
              " 要寫成 :%s.camel —— %s" % (target, ln, attr, _kebab(attr), snippet))
    fails += len(svgc)
    tdz = lint_watch_tdz(text)
    if tdz:
        print(f'[FAIL] {len(tdz)} 處 watch 會讀到還沒宣告的 const '
              f'(watch 立即求值 → TDZ ReferenceError → 整頁白畫面):')
        for wline, name, dline in tdz:
            print(f'       第 {wline} 行的 watch 讀到 {name}，'
                  f'但它在第 {dline} 行才宣告 —— 把 watch 移到宣告之後')
        fails += len(tdz)
    ret_missing = lint_setup_return(text)
    if ret_missing:
        print(f'[FAIL] {len(ret_missing)} 個函式模板有呼叫但 setup 沒回傳 '
              f'(該段模板一旦渲染就白畫面):')
        for name, line in ret_missing:
            print(f'       第 {line} 行 {name}() —— 請加進 setup 的 return')
        fails += len(ret_missing)
    fails += check_inline_js(text, target)
    if fails:
        print(f'[FAIL] {fails} 個問題 — Vue 掛載或 JS 執行會失敗')
    elif sc:
        print(f'[WARN] {sc} 處自閉合可能吞掉後面的節點，請確認該段 UI 有正常顯示')
    else:
        print('[OK] HTML structure parses cleanly')
    return fails, sc


def check_inline_js(text, label):
    """把 inline <script> 丟給 node --check。

    🛑 為什麼非加不可:2026-09-03 加「動態號誌控制」頁時,新函式跟既有的
       TC3 命令下發 loadSigControl 撞名,整份 JS 因 "Identifier has already
       been declared" 直接不執行 —— 連登入表單都動不了。
       但 div 平衡是對的(1654/1654)、模板檢查也全過,兩道既有關卡都看不見它。
       這種錯只有真的把 JS 交給解析器才抓得到。

    node 不在就跳過(只提醒),不要讓沒裝 node 的機器卡住 commit。
    """
    import re as _re
    import shutil
    import subprocess
    import tempfile
    import os

    if not shutil.which('node'):
        print('[SKIP] 找不到 node,跳過 JS 語法檢查'
              '(強烈建議裝上 —— 重複宣告這類錯只有它抓得到)')
        return 0

    blocks = _re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', text, _re.S)
    fails = 0
    for i, b in enumerate(blocks):
        if not b.strip():
            continue
        tmp = tempfile.NamedTemporaryFile('w', suffix='.js', delete=False,
                                          encoding='utf-8')
        tmp.write(b)
        tmp.close()
        try:
            r = subprocess.run(['node', '--check', tmp.name],
                               capture_output=True, text=True)
            if r.returncode != 0:
                fails += 1
                err = (r.stderr or '').strip().splitlines()
                print(f'[FAIL] 第 {i + 1} 個 <script> 語法錯誤 —— '
                      f'整份 JS 不會執行,全站白畫面/無法登入:')
                for line in err[:8]:
                    print('       ' + line)
        finally:
            os.unlink(tmp.name)
    if not fails:
        print('[OK] inline JS 語法通過')
    return fails


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
