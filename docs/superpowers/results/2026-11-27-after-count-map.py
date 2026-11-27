"""After-count row map and blind-referee packet (AuRo test c), mechanical. Labels nothing.

Usage (python3 -I):
  2026-11-27-after-count-map.py DOC COUNTER_LABELS REFEREE_LABELS REFEREE_PACKET TREE_BEFORE TREE_AFTER OUTDIR [SCRATCH]
    DOC            docs/superpowers/results/2026-11-23-patch-constants-baseline.md
    COUNTER_LABELS registry/counter_labels_2026-11-27.tsv
    REFEREE_LABELS registry/referee_labels_2026-11-27.tsv
    REFEREE_PACKET registry/referee_packet_2026-11-27.md      (only its preamble is copied)
    TREE_BEFORE    git archive 535d57ec | tar -x   (common + the nine concept dirs)
    TREE_AFTER     git archive cf65a471 | tar -x
    OUTDIR         where the map / removed / packet files go
    SCRATCH        where ids.tsv goes (default OUTDIR)

Matching (owner's rule, amended by the auro session): an after row takes a frozen label when it has the SAME KEY, the
SAME READER (the loader that reads Section.Key; for a src row the file itself) and the SAME ROLE as a baseline row.
  * same class directory + same key + same reader  -> matched (own row);
  * another class directory, same key + same reader -> INHERITED through the reader (the copy is still its own after row);
  * role is unchanged unless the after-count lists the row as "moved", or the reader's statement for that key changed
    between the two trees (then the row is unmatched and goes to the referee).
Src rows: the baseline row's evidence line (the line within +-6 of the baseline line holding the row's constant name or
value) must occur, whitespace-normalised, in the same file of the after tree; else the row is unmatched/removed.
New src rows: every named constant / numeric literal on lines ADDED between the two trees in in-scope files.
"""
import sys, re, os, collections, subprocess, json, math

DOC, CNT, REF, PKT, TB, TA, OUT = sys.argv[1:8]
SCR = sys.argv[8] if len(sys.argv) > 8 else OUT
TREE_HASH = '5c0029ab522691fb4747d2e7215493162f13f0b0'
CLS_DIR = {'chair': 'chair_concept', 'table': 'table_concept', 'refrigerator': 'refrigerator_concept', 'hood': 'hood_concept',
           'nightstand (config)': 'nightstand_concept', 'bed (config)': 'bed_concept', 'toilet (config)': 'toilet_concept'}
TARGETS = ['chair_concept', 'table_concept', 'refrigerator_concept', 'hood_concept', 'countable_concept']

def rd(root, rel):
    try: return open(os.path.join(root, rel), encoding='utf-8', errors='replace').read()
    except OSError: return None

def norm(s): return re.sub(r'\s+', ' ', s.strip())


def code_part(s):
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s); return re.sub(r'//.*$', '', s)

def code_lines(text):
    """per line: (code with comments and string contents blanked, comment text, code with comments removed and strings kept)."""
    out = []; incm = False
    for ln in text.split('\n'):
        s = ln; code = ''; keep = ''; cm = ''; i = 0; q = None
        while i < len(s):
            two = s[i:i + 2]
            if incm:
                j = s.find('*/', i)
                if j < 0: cm += s[i:]; i = len(s)
                else: cm += s[i:j]; i = j + 2; incm = False
                continue
            ch = s[i]
            if q:
                if ch == '\\': i += 2; code += '  '; keep += s[i - 2:i]; continue
                if ch == q: q = None; code += ch
                else: code += ' '
                keep += ch; i += 1; continue
            if two == '//': cm += s[i + 2:]; break
            if two == '/*': incm = True; i += 2; continue
            if ch in '"\'': q = ch; code += ch; keep += ch; i += 1; continue
            code += ch; keep += ch; i += 1
        out.append((code, cm.strip(), keep))
    return out

_FN_CACHE = {}
CTRL = re.compile(r'^\s*(?:\}\s*)?(?:else\s+)?(?:if|for|while|switch|catch|do|else|try)\b')
def functions(root, rel):
    """outermost function bodies of a C++ file: list of dict(sig, start, end, body). Heuristic brace scan (no parser)."""
    key = (root, rel)
    if key in _FN_CACHE: return _FN_CACHE[key]
    t = rd(root, rel)
    res = []
    if t is not None:
        cl = code_lines(t); stack = []; pre = ''; pre_line = None
        for i, (c, cm, keep) in enumerate(cl):
            ln = i + 1
            if c.strip().startswith('#'): pre = ''; pre_line = None; continue
            for ch in c:
                if ch in ';{}':
                    if ch == '{':
                        in_fn = any(k == 'func' for k, _, _ in stack)
                        kind = 'inner'
                        if not in_fn:
                            ps = pre.strip()
                            core = re.sub(r'(?:\bconst\b|\bnoexcept\b(?:\([^)]*\))?|\boverride\b|\bfinal\b|\bmutable\b|\bvolatile\b|&&?|->\s*[\w:<>,\s\*&]+?|\brequires\b[^)]*\))\s*$', '', ps).strip()
                            if re.search(r'\b(?:namespace|class|struct|union|enum)\b', ps) and not ps.endswith(')'): kind = 'other'
                            elif ps.endswith('=') or ps == '': kind = 'other'
                            elif core.endswith(')') or ps.endswith(')'): kind = 'func'
                            else: kind = 'other'
                        stack.append((kind, pre_line or ln, norm(pre)))
                    elif ch == '}':
                        if stack:
                            kind, st, sig = stack.pop()
                            if kind == 'func' and not any(k == 'func' for k, _, _ in stack):
                                body = [norm(x[2]) for x in cl[st - 1:ln] if x[2].strip()]
                                res.append(dict(sig=sig, start=st, end=ln, body=body))
                    pre = ''; pre_line = None
                else:
                    if pre_line is None and not ch.isspace(): pre_line = ln
                    pre += ch
            pre += ' '
    _FN_CACHE[key] = res
    return res
def enclosing(root, rel, line):
    for fn in functions(root, rel):
        if fn['start'] <= line <= fn['end']: return fn
    return None
def fn_text(root, rel, fn):
    ls = rd(root, rel).split('\n'); return ls[fn['start'] - 1:fn['end']]

# ------------------------------------------------------------------ 1. baseline rows + ids
L = open(DOC, encoding='utf-8').read().split('\n')
end = next(i for i, l in enumerate(L) if l.startswith('## After part 1'))
BL = L[:end]
targets_files = {}; cur = None; d = None; inblk = False
for l in BL:
    m = re.match(r'<details><summary>(\w+): \d+ files</summary>', l)
    if m: cur = m.group(1); targets_files[cur] = set(); continue
    if l.startswith('### Newly counted files'): cur = None
    if cur is None: continue
    m = re.match(r'\*\*(.+?/)\*\* \(\d+\)', l)
    if m: d = m.group(1); continue
    if l.startswith('```'): inblk = not inblk; continue
    if inblk and l.strip(): targets_files[cur].add(d + l.strip())

def brace(s):
    m = re.search(r'\{([^{}]*)\}', s)
    if not m: return [s]
    return [x for alt in m.group(1).split(',') for x in brace(s[:m.start()] + alt + s[m.end():])]
THREE = ['common/concept_manifest/manifest_toml.h', 'common/standpoint/residual_risk.h', 'common/standpoint/standpoint_oracle.h']
FR = THREE + brace('common/ai_belief/{beam_channel,beam_room,box_engine_model,foreign_box,offer_view,persistent_engine,persistent_update,viewpoint}.h') + ['common/place_memory/place_map.h']
CT = THREE + brace('common/ai_belief/{beam_channel,foreign_box,offer_view,parts_geometry,persistent_engine,viewpoint}.h') \
    + brace('common/appearance_belief/{appearance_belief,colour_name,colour_terms}.h') \
    + brace('common/concept_manifest/{layer_spread,permanence,wall_standoff}.h') + ['common/epistemic_step/offer_cache.h', 'common/recall_match/identity_link.h'] \
    + brace('common/thumbs/{thumb_select,thumb_wire}.h') + brace('countable_concept/src/countable_{birth_seed,gain_query,latent_inputs,neighbour_size,replay,size_std,vertical_span}.h') \
    + ['countable_concept/src/specificworker_consumers.cpp']
SCOPE = {'chair_concept': set(targets_files['chair_concept']) | set(THREE), 'table_concept': set(targets_files['table_concept']) | set(THREE),
         'refrigerator_concept': set(targets_files['refrigerator_concept']) | set(FR), 'hood_concept': set(targets_files['hood_concept']) | set(FR),
         'countable_concept': (set(targets_files['countable_concept']) | set(CT)) - {'countable_concept/src/countable_viewpoint.cpp'}}
SCOPE_MISSING = {t: sorted(f for f in s if rd(TA, f) is None) for t, s in SCOPE.items()}
for t in SCOPE: SCOPE[t] = {f for f in SCOPE[t] if rd(TA, f) is not None}
def targets_of(f): return [t for t in TARGETS if f in SCOPE[t]]

def split_row(l): return [x.strip() for x in re.split(r'(?<!\\)\|', l.strip().strip('|'))]
WHERE_RE = re.compile(r'([\w][\w/.\-]*\.(?:h|cpp|hpp))(?::(\d+))?')
base = []
h2 = h3 = h4 = None; hdr = None
for i, l in enumerate(BL):
    if l.startswith('## '): h2 = l[3:].strip(); h3 = h4 = None; hdr = None; continue
    if l.startswith('### '): h3 = l[4:].strip(); h4 = None; hdr = None; continue
    if l.startswith('#### '): h4 = l[5:].strip(); hdr = None; continue
    if not l.startswith('|'): hdr = None; continue
    c = split_row(l)
    if (c[0] in ('name', 'class') and c[-1] in ('label', 'kind')):
        hdr = c; continue
    if l.startswith('|---') or hdr is None: continue
    own = hdr[0] == 'class'
    cls = c[0] if own else None
    name, value, where, role, lab = (c[1], c[2], c[3], c[4], c[5]) if own else (c[0], c[1], c[2], c[3], c[4])
    if hdr[-1] == 'kind': lab = 'switch (patch)' if lab == 'patch mechanism' else 'switch (other)'
    sec = ' / '.join(x for x in (h2, h3, h4) if x)
    base.append(dict(line=i + 1, h2=h2, sec=sec, name=name, value=value, where=where, role=role, label=lab, cls=cls,
                     best=c[-1] if hdr[-1] == 'label' else lab, kindtab=hdr[-1]))
# ? rows align with the counter ids (same rule as the addendum 3 script)
cnt = [l.rstrip('\n').split('\t') for l in open(CNT, encoding='utf-8')]
ref = {}
for l in open(REF, encoding='utf-8'):
    if l.startswith('#') or l.startswith('id\t'): continue
    p = l.rstrip('\n').split('\t'); ref[p[0]] = p[1]
q = [r for r in base if r['kindtab'] == 'label' and r['best'].endswith('?') and not r['best'].startswith('ambig')]
bad = 0
for r, c in zip(q, cnt):
    if not (c[1] == r['sec'] and c[3] == r['best']): bad += 1
    r['id'] = c[0]; r['frozen'] = ref[c[0]]
assert len(q) == len(cnt) == len(ref) == 340 and bad == 0, (len(q), len(cnt), len(ref), bad)
n = 0
for r in base:
    if 'id' not in r:
        n += 1; r['id'] = 'B%04d' % n; r['frozen'] = r['label']
# kind, file, reader, home class dir
for r in base:
    w = r['where']
    r['kind'] = 'src'; r['reader'] = None; r['file'] = None; r['home'] = None; r['line_no'] = None
    if r['h2'] in CLS_DIR and (w.startswith('etc/config.toml') or w.startswith('NO toml key')):
        home = CLS_DIR[r['h2']]
        r['home'] = home
        m = re.search(r'(?:read|code default at) ([\w/.]+\.cpp)(?::(\d+))?', w)
        r['reader'] = os.path.basename(m.group(1)) if m else 'plumbing'
        r['kind'] = 'hidden' if w.startswith('NO toml key') else 'config'
        r['file'] = home + '/etc/config.toml'
        r['line_no'] = int(m.group(2)) if m and m.group(2) else None
    else:
        m = WHERE_RE.search(w)
        if m: r['file'] = m.group(1); r['line_no'] = int(m.group(2)) if m.group(2) else None
        elif r['cls']: r['file'] = None
with open(os.path.join(SCR, 'ids.tsv'), 'w', encoding='utf-8') as f:
    f.write('id\tsection\tkey\tvalue\twhere\tbaseline_label\tfrozen_label\tkind\tfile\treader\n')
    for r in base:
        f.write('\t'.join([r['id'], r['sec'], r['name'], r['value'], r['where'], r['label'], r['frozen'], r['kind'], r['file'] or '', r['reader'] or '']) + '\n')

# ------------------------------------------------------------------ 2. after config rows
def parse_toml(text):
    out = []; sec = ''; lines = text.split('\n'); i = 0
    while i < len(lines):
        ln = lines[i]; s = ln.strip(); i += 1
        if not s or s.startswith('#'): continue
        m = re.match(r'^\[\[?([^\]]+)\]\]?', s)
        if m and '=' not in s.split(']')[0]: sec = m.group(1).strip(); continue
        m = re.match(r'^([A-Za-z0-9_\-."]+)\s*=\s*(.*)$', s)
        if not m: continue
        key = m.group(1).strip('"'); val = m.group(2); startline = i
        def strip_c(v):
            res = []; q = None
            for ch in v:
                if q:
                    res.append(ch)
                    if ch == q: q = None
                elif ch in '"\'': q = ch; res.append(ch)
                elif ch == '#': break
                else: res.append(ch)
            return ''.join(res).strip()
        val = strip_c(val)
        while val.count('[') > val.count(']') and i < len(lines):
            val += ' ' + strip_c(lines[i]); i += 1
        out.append((sec + '.' + key if sec else key, val, startline))
    return out

KEYCALL = re.compile(r'\b[A-Za-z_]\w*\(\s*"([A-Z]\w*(?:\.\w+)+)"')
def loader_stmts(root, rel):
    """key -> list of (line, default text, normalised statement) for the loader file."""
    t = rd(root, rel)
    if t is None: return {}
    res = collections.defaultdict(list)
    for m in KEYCALL.finditer(t):
        key = m.group(1); ln = t.count('\n', 0, m.start()) + 1
        j = m.end(); q = False; depth = 1; stmt_end = None; k = j
        # default: text after the key literal up to the next top-level ',' or ')'
        dstart = j + 1; dtxt = ''
        # statement: to the next ';' outside quotes
        qq = None
        for k in range(m.start(), len(t)):
            ch = t[k]
            if qq:
                if ch == '\\': continue
                if ch == qq and t[k - 1] != '\\': qq = None
            elif ch in '"\'': qq = ch
            elif ch == ';': stmt_end = k; break
        sstart = max(t.rfind(ch_, 0, m.start()) for ch_ in ';{}') + 1
        stmt = t[sstart:stmt_end if stmt_end else m.end()]
        rest = t[m.end():]
        mm = re.match(r'\s*,\s*', rest)
        what = ''
        if mm:
            p = mm.end(); args = []; dep = 0; qq = None; buf = ''
            while p < len(rest):
                ch = rest[p]
                if qq:
                    buf += ch
                    if ch == qq and rest[p - 1] != '\\': qq = None
                elif ch in '"\'': qq = ch; buf += ch
                elif ch in '([{': dep += 1; buf += ch
                elif ch in ')]}':
                    if dep == 0: break
                    dep -= 1; buf += ch
                elif ch == ',' and dep == 0: args.append(buf); buf = ''
                else: buf += ch
                p += 1
            args.append(buf)
            dtxt = norm(args[0]) if args else ''
            what = ' '.join(re.findall(r'"((?:[^"\\]|\\.)*)"', args[1])) if len(args) > 1 else ''
        res[key].append((ln, dtxt, norm(stmt), norm(what)))
    return res

LOADER = {'chair_concept': 'chair_concept/src/chair_config.cpp', 'table_concept': 'table_concept/src/table_config.cpp',
          'refrigerator_concept': 'refrigerator_concept/src/refrigerator_config.cpp', 'hood_concept': 'hood_concept/src/hood_config.cpp',
          'countable': 'countable_concept/src/countable_config.cpp'}
AFTER_TOMLS = []  # (toml path, class dir, loader key)
for cd in ['chair_concept', 'table_concept', 'refrigerator_concept', 'hood_concept']:
    AFTER_TOMLS.append((cd + '/etc/config.toml', cd, cd)); AFTER_TOMLS.append((cd + '/etc/countable.toml', cd, 'countable'))
for cd in ['nightstand_concept', 'bed_concept', 'toilet_concept']: AFTER_TOMLS.append((cd + '/etc/config.toml', cd, 'countable'))
AFTER_TOMLS = [x for x in AFTER_TOMLS if rd(TA, x[0]) is not None]
LA = {k: loader_stmts(TA, v) for k, v in LOADER.items()}
LB = {k: loader_stmts(TB, v) for k, v in LOADER.items()}

def toml_comments(text, startline):
    """the comment block directly above line `startline` (1-based) + the trailing comment of that line."""
    ls = text.split('\n'); out = []; i = startline - 2
    while i >= 0 and ls[i].strip().startswith('#') and len(out) < 4:
        out.insert(0, ls[i].strip().lstrip('#').strip()); i -= 1
    cur = ls[startline - 1]; m = re.search(r'\s#\s?(.*)$', cur)
    tail = m.group(1).strip() if m else ''
    return ' '.join(x for x in out if x), tail

after = []   # dict rows
def add_after(**kw):
    kw.setdefault('inherited', ''); kw.setdefault('note', ''); kw.setdefault('mid', ''); kw.setdefault('frozen', ''); kw.setdefault('role', '')
    after.append(kw); return kw

for toml, cd, lk in AFTER_TOMLS:
    text = rd(TA, toml); rows = parse_toml(text); ld = LA[lk]; lpath = LOADER[lk]
    tkeys = set()
    tgt = ['countable_concept'] if lk == 'countable' else [cd]
    for key, val, ln in rows:
        tkeys.add(key)
        st = ld.get(key)
        reader = os.path.basename(lpath) if st else 'plumbing'
        above, tail = toml_comments(text, ln)
        what = ''
        role = ' '.join(x for x in [above, tail] if x)
        if st and st[0][3]: role = (role + ' | ' if role else '') + 'loader: ' + st[0][3]
        add_after(kind='config', key=key, value=val, where='%s:%d [%s]%s' % (toml, ln, key.rsplit('.', 1)[0], ('; read %s:%d' % (lpath, st[0][0])) if st else ''),
                  file=toml, cd=cd, reader=reader, tgt=tgt, role=role, loader=lk)
    for key, st in ld.items():
        if key in tkeys: continue
        ln, dtxt, _, what = st[0]
        add_after(kind='hidden', key=key, value=dtxt, where='NO toml key; code default at %s:%d' % (lpath, ln), file=toml, cd=cd,
                  reader=os.path.basename(lpath), tgt=tgt, role=('loader: ' + what) if what else '(loader default, no toml key; no description string)', loader=lk)

# --- moved rows named by the after-count
MOVED_CFG_SUFFIX = ('AI2CommonModePosStd', 'AI2CommonModeSizeStd', 'AI2CommonModeYawStd')   # countable path, patch -> sensor?
def is_moved_cfg(key, reader): return reader == 'countable_config.cpp' and key.endswith(MOVED_CFG_SUFFIX)

CONSUMER_NOTE = {}
def consumer_lines(root, files, field):
    c = collections.Counter()
    pat = re.compile(r'\b' + re.escape(field) + r'\b')
    for f in files:
        if not f.endswith(('.h', '.hpp', '.cpp')): continue
        t = rd(root, f)
        if t is None: continue
        for x in t.split('\n'):
            cp = code_part(x)
            if pat.search(cp): c[norm(cp)] += 1
    return c
def loader_changed(loaderkey, key):
    """'' = unchanged (reading line unchanged, or description-only edit with every consumer unchanged); else the reason."""
    bs = LB[loaderkey].get(key, []); as_ = LA[loaderkey].get(key, [])
    b = {x[2] for x in bs}; a = {x[2] for x in as_}
    if not bs: return None
    if a == b: return ''
    strip = lambda t: re.sub(r'"(?:[^"\\]|\\.)*"', '""', t)
    if {strip(x) for x in a} != {strip(x) for x in b}: return 'reading line changed (default/read call)'
    # description-only edit of the reading line: the role is judged by use. Every consumer of the value must be unchanged.
    m = re.match(r'\s*([\w.\[\]>-]+)\s*=\s*[\w:]+\(\s*"', next(iter(a)))
    if not m:
        m2 = re.search(r'([\w.]+)\s*=\s*(?:\w+\()?\s*$', '')
        return 'description-only edit, consumer field not identifiable'
    field = re.split(r'[.>-]+', m.group(1))[-1]
    tgt = 'countable_concept' if loaderkey == 'countable' else loaderkey
    files = sorted(SCOPE[tgt] | {LOADER[loaderkey].replace('.cpp', '.h')})
    cb = consumer_lines(TB, files, field); ca = consumer_lines(TA, files, field)
    CONSUMER_NOTE[(loaderkey, key)] = 'field %s: %d consumer lines before, %d after' % (field, sum(cb.values()), sum(ca.values()))
    if cb == ca: return ''
    diff = (ca - cb) + (cb - ca)
    return 'description-only edit but a consumer of %s changed (%d lines differ)' % (field, sum(diff.values()))

LOADER_OF_READER = {os.path.basename(v): k for k, v in LOADER.items()}
bycfg = collections.defaultdict(list)   # (key, reader) -> baseline rows
READER_FIX = 0
for r in base:
    if r['kind'] in ('config', 'hidden'):
        # the reader is where the key is quoted in the class's loader at the BEFORE tree (the doc text omits it for some rows)
        lk = 'countable' if r['home'] in ('nightstand_concept', 'bed_concept', 'toilet_concept') else r['home']
        rd_ = os.path.basename(LOADER[lk]) if r['name'] in LB[lk] else 'plumbing'
        if rd_ != r['reader']: READER_FIX += 1
        r['reader'] = rd_
        bycfg[(r['name'], r['reader'])].append(r)
matched_ids = set(); removed = {}   # baseline id -> reason
moved_ids = {}

for a in [x for x in after if x['kind'] in ('config', 'hidden')]:
    cands = bycfg.get((a['key'], a['reader']), [])
    if not cands: a['note'] = 'no baseline row with this key and reader'; continue
    own = [c for c in cands if c['file'] == a['file']]
    pool = own if own else cands
    # role unchanged?
    ok = []
    chg = ''
    for c in pool:
        if is_moved_cfg(a['key'], a['reader']): chg = 'moved (after-count: patch -> sensor?)'; moved_ids[c['id']] = chg; continue
        lc = loader_changed(LOADER_OF_READER[a['reader']], a['key']) if a['reader'] in LOADER_OF_READER else ''
        if lc: chg = 'loader statement changed: ' + lc; moved_ids[c['id']] = chg; continue
        ok.append(c)
    if not ok:
        a['note'] = chg; continue
    c = ok[0]
    a['mid'] = c['id']; a['frozen'] = c['frozen']
    if not own:
        a['inherited'] = 'Y'; labs = {x['frozen'] for x in ok}
        a['note'] = 'inherited via reader %s from %s' % (a['reader'], ','.join(x['id'] for x in ok)) + ('' if len(labs) == 1 else '; CANDIDATE LABELS DISAGREE: ' + '/'.join(sorted(labs)))
    else: matched_ids.add(c['id'])
# an own baseline row is matched only through an own after row (computed above); others are removed
# ------------------------------------------------------------------ 3. baseline src rows relocated
SRC_MOVED = {('countable_concept/src/countable_belief.h', 'gn_tol_sigma'): 'moved (after-count: excluded -> patch)',
             ('countable_concept/src/countable_belief.cpp', 'wall-hit count margin'): 'moved to common/ai_belief/beam_channel.h:83 (after-count)'}
NUM = re.compile(r'(?<![\w.])(\d+\.\d*|\.\d+|\d+)(?:[eE]([+-]?\d+))?[fFlLuU]*(?![\w])')
def nums_of(s):
    out = []
    for m in NUM.finditer(s):
        try: out.append(float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1))
        except ValueError: pass
    return out
unlocated = []
src_after_pos = {}   # (file,line) -> after row, for the diff scan to avoid duplicates
for r in base:
    if r['kind'] != 'src': continue
    f = r['file']; reason = None
    key = (f, r['name'])
    if key in SRC_MOVED: removed[r['id']] = SRC_MOVED[key]; continue
    if not f: unlocated.append((r, 'no file in the where column')); continue
    bt = rd(TB, f); at = rd(TA, f)
    if bt is None and rd(TA, f) is None: unlocated.append((r, 'file not in either tree')); continue
    if at is None: removed[r['id']] = 'file gone from the after tree'; continue
    if bt is None: unlocated.append((r, 'file absent from the before tree')); continue
    bl = bt.split('\n'); L0 = r['line_no']
    if not L0: unlocated.append((r, 'no line in the where column')); continue
    idt = [t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', r['name']) if len(t) >= 3 and re.search(r'[A-Z_0-9]', t[1:] + t[:0])]
    vn = nums_of(r['value'])
    best = None
    for j in range(max(0, L0 - 7), min(len(bl), L0 + 6)):
        cp = code_part(bl[j])
        if not cp.strip(): continue
        sc = 2 * sum(1 for t in idt if re.search(r'\b' + re.escape(t) + r'\b', cp)) + sum(1 for x in nums_of(cp) if any(abs(x - y) <= 1e-12 + 1e-6 * abs(y) for y in vn))
        if sc and (best is None or sc > best[0] or (sc == best[0] and abs(j + 1 - L0) < abs(best[1] + 1 - L0))): best = (sc, j)
    if best is None: unlocated.append((r, 'no line within +-6 of %s:%d carries the row name or value' % (f, L0))); continue
    ev = norm(code_part(bl[best[1]]))
    al = at.split('\n'); occ = [i + 1 for i, x in enumerate(al) if norm(code_part(x)) == ev]
    if not occ: removed[r['id']] = 'evidence line text changed or gone (%s:%d: %s)' % (f, best[1] + 1, ev[:90]); continue
    nl = min(occ, key=lambda x: abs(x - (best[1] + 1)))
    # the enclosing function must be unchanged too (same signature, same body without comments)
    fb = enclosing(TB, f, best[1] + 1); fa = enclosing(TA, f, nl)
    ctx = None
    if (fb is None) != (fa is None) or (fb and (fb['sig'] != fa['sig'] or fb['body'] != fa['body'])):
        import difflib
        bt_l = fn_text(TB, f, fb) if fb else []; at_l = fn_text(TA, f, fa) if fa else []
        sm = difflib.SequenceMatcher(None, [norm(x) for x in bt_l], [norm(x) for x in at_l], autojunk=False)
        hunks = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != 'equal': hunks.append((tag, (fb['start'] if fb else 1) + i1, bt_l[i1:i2], (fa['start'] if fa else 1) + j1, at_l[j1:j2]))
        ctx = dict(sig=(fa or fb)['sig'][:160], bstart=fb['start'] if fb else None, astart=fa['start'] if fa else None, hunks=hunks,
                   ev_before='%s:%d' % (f, best[1] + 1), ev_text=norm(bl[best[1]]), ev_after='%s:%d' % (f, nl),
                   fn_lines=(len(bt_l), len(at_l)))
    a = add_after(kind='src', key=r['name'], value=r['value'], where='%s:%d' % (f, nl), file=f, cd=None, reader=f, tgt=targets_of(f) or [], role='',
                  mid='' if ctx else r['id'], frozen='' if ctx else r['frozen'], base_role=r['role'])
    if ctx:
        a['ctx'] = ctx; a['note'] = 'context changed (enclosing function %s differs between the trees); baseline %s' % (ctx['sig'][:60], r['id'])
        removed[r['id']] = 'context changed: enclosing function of the evidence line changed (%s)' % ctx['sig'][:80]
    else: matched_ids.add(r['id'])
    src_after_pos[(f, nl)] = a

# ------------------------------------------------------------------ 4. diff scan of added lines
PRINT = re.compile(r'std::print|println|printf|qDebug|qInfo|qWarning|qCritical|std::cout|std::cerr|std::format|fprintf|snprintf|\bcheck\(|\bCHECK|\bassert\(|static_assert|\bthrow\b|std::clog|\blog_\w*\(|\bLOG\w*\(')
NAMED = re.compile(r'\b(?:constexpr|static\s+const|static\s+constexpr|inline\s+constexpr)\b[^=;(){}]*?\b([A-Za-z_]\w*)\s*(?:=|\{)\s*([^;]*)')
KNAME = re.compile(r'\b(k[A-Z]\w*)\s*(?:=|\{)\s*([^;,]*)')
DECL = re.compile(r'\b([A-Za-z_]\w*)\s*(?:=|\{)\s*[-+(]*\s*\d')
def added_lines(f):
    at = rd(TA, f); bt = rd(TB, f)
    n = at.count('\n') + 1
    if bt is None: return set(range(1, n + 1))
    p = subprocess.run(['diff', '--old-line-format=', '--unchanged-line-format=', '--new-line-format=%dn\n',
                        os.path.join(TB, f), os.path.join(TA, f)], capture_output=True, text=True)
    return {int(x) for x in p.stdout.split() if x.isdigit()}

def trivial(x, raw):
    return x in (0, 0.5, 1, 1.5, 2) or re.match(r'^1(\.0*)?[eE]-\d+[fFlL]*$', raw)

scan_rows = []
scope_files = sorted(set().union(*SCOPE.values()))
for f in scope_files:
    if not f.endswith(('.h', '.hpp', '.cpp', '.cc')): continue
    if '/tests/' in f or 'test' in os.path.basename(f) or 'selftest' in f: continue
    at = rd(TA, f); add = added_lines(f)
    if not add: continue
    cl = code_lines(at)
    # statements: a statement starts after a line ending in ; { } or after a blank / preprocessor line
    stmt_id = []; sid = 0; prev_end = True
    for i, (c, cm, _k) in enumerate(cl):
        s = c.strip()
        if prev_end: sid += 1
        stmt_id.append(sid)
        prev_end = (not s) or s.startswith('#') or s.endswith((';', '{', '}', ':')) or s.endswith('*/')
    stmt_print = collections.defaultdict(bool)
    for i, (c, cm, _k) in enumerate(cl):
        if PRINT.search(c): stmt_print[stmt_id[i]] = True
        if KEYCALL.search(c): stmt_print[stmt_id[i]] = True     # a loader read: its constant is a config row
    for i, (c, cm, _k) in enumerate(cl):
        ln = i + 1
        if ln not in add: continue
        s = c.strip()
        if not s or s.startswith('#') or stmt_print[stmt_id[i]]: continue
        if KEYCALL.search(c): continue        # loader read: a config row
        m = NAMED.search(c) or KNAME.search(c)
        nums = []
        for mm in NUM.finditer(c):
            raw = mm.group(0)
            try: v = float(mm.group(1)) * (10 ** int(mm.group(2)) if mm.group(2) else 1)
            except ValueError: continue
            st = mm.start()
            if c[st - 1:st] == '[' and c[mm.end():mm.end() + 1] == ']': continue      # array index / size
            if re.search(r'[A-Za-z_:]\w*<[^<>()=!&|;]*$', c[:st]) and re.match(r'[^<>()=!&|;]*>', c[mm.end():]): continue   # template argument
            if trivial(v, raw) and not m: continue
            nums.append(raw)
        if not nums: continue
        if m and not [x for x in nums_of(m.group(2) if m.lastindex and m.lastindex >= 2 else '')] and not any(True for _ in NUM.finditer(m.group(0))):
            continue
        key = None
        if m: key = m.group(1)
        else:
            dm = DECL.search(c); key = dm.group(1) if dm else 'literal'
        above = []; j = i - 1
        while j >= 0 and not cl[j][0].strip() and cl[j][1] and len(above) < 3: above.insert(0, cl[j][1]); j -= 1
        role = ' / '.join(x for x in above + [cm] if x)
        scan_rows.append(dict(kind='src', key=key, value=', '.join(nums) if not m else norm(m.group(2))[:80] or ', '.join(nums), where='%s:%d' % (f, ln), file=f,
                              cd=None, reader=f, tgt=targets_of(f), role=role, named=bool(m)))
for s in scan_rows:
    if (s['file'], int(s['where'].rsplit(':', 1)[1])) in src_after_pos: continue   # a relocated baseline row already sits there
    add_after(**s)

# ------------------------------------------------------------------ 5. after-count's own added rows (doc "Rows added by part 1")
ac_start = next(i for i, l in enumerate(L) if l.startswith('### Rows added by part 1'))
ac_end = next(i for i, l in enumerate(L) if l.startswith('### Combined totals per class after part 1'))
AC = []
for l in L[ac_start:ac_end]:
    if not l.startswith('|') or l.startswith('|---'): continue
    c = split_row(l)
    if c[0] in ('row', 'name') or len(c) < 3: continue
    if re.search(r"NOT a new row|the baseline's row", ' '.join(c)): continue
    AC.append(c)
by_pos = {(a['file'], int(a['where'].rsplit(':', 1)[1])): a for a in after if a['kind'] == 'src'}
by_file_base = collections.defaultdict(list)
for f in scope_files: by_file_base[os.path.basename(f)].append(f)
ac_added = []
for c in AC:
    nm, val, wh = c[0], c[1], c[2]
    note = c[3] if len(c) > 3 else ''
    wh2 = wh.replace('`', '')
    m = re.search(r'([\w/.]+\.(?:h|cpp))(?::(\d+))?', wh2)
    f = ln = None
    if m:
        cand = [x for x in scope_files if x == m.group(1) or x.endswith('/' + m.group(1))]
        f = cand[0] if len(cand) == 1 else None; ln = int(m.group(2)) if m.group(2) else None
    hit = by_pos.get((f, ln)) if f and ln else None
    cfg_hits = [a for a in after if a['kind'] in ('config', 'hidden') and f and ln and re.search(re.escape('%s:%d' % (f, ln)) + r'(?!\d)', a['where'])]
    if cfg_hits:
        for a in cfg_hits: a['note'] = (a['note'] + '; ' if a['note'] else '') + 'after-count row: ' + re.sub(r'`', '', nm)
        continue
    if hit: hit['note'] = (hit['note'] + '; ' if hit['note'] else '') + 'after-count row: ' + re.sub(r'`', '', nm); continue
    ac_added.append((nm, val, wh2, f, ln, note))
for nm, val, wh2, f, ln, note in ac_added:
    code = ''
    if f and ln:
        al = rd(TA, f).split('\n'); code = norm(al[ln - 1]) if ln <= len(al) else ''
    add_after(kind='src', key=re.sub(r'[`*]', '', nm), value=re.sub(r'[`*]', '', val), where=(f + ':%d' % ln) if f and ln else wh2, file=f, cd=None, reader=f,
              tgt=targets_of(f) if f else [], role=('code: ' + code[:200]) if code else '(location not a single file:line in the doc)', note='after-count row (doc) not found on an added line')

# ------------------------------------------------------------------ 5b. the unresolved baseline rows go to the packet (auro ruling)
_CL = {}
def code_role(root, f, ln):
    """the code's own text at file:line: the comment lines directly above, the trailing comment, and the line itself."""
    if (root, f) not in _CL: _CL[(root, f)] = code_lines(rd(root, f) or '')
    cl = _CL[(root, f)]
    if not (1 <= ln <= len(cl)): return ''
    above = []; j = ln - 2
    while j >= 0 and not cl[j][0].strip() and cl[j][1] and len(above) < 3: above.insert(0, cl[j][1]); j -= 1
    return ' / '.join(x for x in above + [cl[ln - 1][1]] if x)
for r, why in unlocated:
    f = r['file']; ln = r['line_no']
    txt = ''
    if f and ln and rd(TB, f) is not None:
        bl_ = rd(TB, f).split('\n'); txt = ' ; '.join(norm(x) for x in bl_[max(0, ln - 2):ln + 1] if x.strip())
    a = add_after(kind='src', key=r['name'], value=r['value'], where=('%s:%s [BASELINE tree 535d57ec, not found by name or value]' % (f, ln)) if f else r['where'],
                  file=f, cd=None, reader=f, tgt=targets_of(f) if f else [], role='', note='moved or removed?; baseline %s; %s' % (r['id'], why))
    a['moved_or_removed'] = dict(text=txt, comment=code_role(TB, f, ln) if f and ln else '')

# ------------------------------------------------------------------ 6. ids, outputs
for k, a in enumerate(after): a['aid'] = 'A%04d' % (k + 1)
def tg(a): return '+'.join(a['tgt']) if a['tgt'] else '(none)'
for a in after:
    if a['kind'] in ('config', 'hidden') and a['loader'] == 'countable' and a['cd'] not in ('nightstand_concept', 'bed_concept', 'toilet_concept'):
        pass
UNREAD_FILE = 'table_concept/etc/countable.toml'   # see report: opened by no table_concept code path
for a in after:
    if a.get('file') == UNREAD_FILE and a['kind'] in ('config', 'hidden'):
        a['unread'] = True; a['frozen'] = 'excluded'
        a['note'] = 'present but unread (rule 0): ' + ('was ' + a['note'] + '; ' if a['note'] else '') + 'no table_concept code path opens etc/countable.toml'
        a['inherited'] = ''
unread = [a for a in after if a.get('unread')]
unmatched = [a for a in after if not a['mid'] and not a.get('unread')]
matched = [a for a in after if a['mid'] and not a.get('unread')]
for r in base:
    if r['id'] in matched_ids or r['id'] in removed: continue
    if r['id'] in moved_ids: removed[r['id']] = 'role moved / reader text changed: ' + moved_ids[r['id']]; continue
    if any(x is r for x, _ in unlocated): continue
    if r['id'] in removed: continue
    if r['kind'] in ('config', 'hidden'):
        own = [a for a in after if a['kind'] in ('config', 'hidden') and a['cd'] == r['home'] and a['key'] == r['name']]
        removed[r['id']] = ('key now has another reader (%s)' % ','.join(sorted({a['reader'] for a in own}))) if own else 'key gone from the class tomls and from the loaders'
    else: removed[r['id']] = 'not relocated'
os.makedirs(OUT, exist_ok=True)
def esc(s): return str(s).replace('\t', ' ').replace('\n', ' ')
with open(os.path.join(OUT, '2026-11-27-after-count-map.tsv'), 'w', encoding='utf-8') as f:
    f.write('after_row_id\ttarget(s)\tkey\tvalue\twhere\tmatched_baseline_id\tfrozen_label\tinherited_via_reader\tnote\n')
    for a in after:
        f.write('\t'.join(esc(x) for x in [a['aid'], tg(a), a['key'], a['value'], a['where'], a['mid'], a['frozen'], a['inherited'], a['note']]) + '\n')
byid = {r['id']: r for r in base}
with open(os.path.join(OUT, '2026-11-27-after-count-removed.tsv'), 'w', encoding='utf-8') as f:
    f.write('baseline_id\tsection\tkey\twhere\tbaseline_label\tfrozen_label\twhy\n')
    for r in base:
        if r['id'] in removed: f.write('\t'.join(esc(x) for x in [r['id'], r['sec'], r['name'], r['where'], r['label'], r['frozen'], removed[r['id']]]) + '\n')
    for r, why in unlocated:
        f.write('\t'.join(esc(x) for x in [r['id'], r['sec'], r['name'], r['where'], r['label'], r['frozen'], 'UNRESOLVED, now in the packet as "moved or removed?" (not counted removed): ' + why]) + '\n')

# packet
pk = open(PKT, encoding='utf-8').read()
pre = pk[pk.index('## Definition'):pk.index('## Rows to label')].rstrip() + '\n'
rules = []
for l in open(REF, encoding='utf-8'):
    if l.startswith('# Rules applied') or (rules and l.startswith('#') and not l.startswith('# Totals')): rules.append(l.rstrip('\n'))
    elif rules: break
def qrole(a):
    if a.get('moved_or_removed'):
        m = a['moved_or_removed']
        return '**moved or removed?** baseline location %s; baseline code: "%s"%s; after tree %s' % (a['where'].split(' [')[0], m['text'][:260].replace('"', "'"), ('; comment: "' + m['comment'] + '"') if m['comment'] else '', TREE_HASH)
    if a.get('ctx'):
        return '**context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: %s. Code comment: %s' % (a['key'], ('"' + (code_role(TA, a['file'], int(a['where'].rsplit(':', 1)[1])) or '') + '"') if code_role(TA, a['file'], int(a['where'].rsplit(':', 1)[1])) else '(none)')
    t = a['role'] or ''
    t = re.sub(r'\s+', ' ', t).strip()
    return ('"' + t[:320] + ('…' if len(t) > 320 else '') + '"') if t else '(no comment in the code)'
def loc_file(a): return a['file'] if a['file'] else '(unlocated)'
byfile = collections.defaultdict(list)
for a in unmatched: byfile[loc_file(a)].append(a)
def lineno(a):
    m = re.search(r':(\d+)', a['where']); return int(m.group(1)) if m else 0
def ctx_block(a):
    c = a['ctx']; o = ['\n<details><summary>%s: context changed, function `%s` (%d lines before, %d after)</summary>\n' % (a['aid'], c['sig'].replace('`', "'"), c['fn_lines'][0], c['fn_lines'][1]),
                    'Evidence line (before, %s): `%s`; found unchanged at %s.\n' % (c['ev_before'], c['ev_text'].replace('`', "'"), c['ev_after'])]
    budget = 80
    for tag, bl, bt_, al, at_ in c['hunks']:
        take = lambda xs: xs[:max(0, budget)]
        o.append('```\nBEFORE (from line %d)\n%s%s\nAFTER (from line %d)\n%s%s\n```' % (bl, '\n'.join(take(bt_)), ('\n... (%d more lines)' % (len(bt_) - budget)) if len(bt_) > budget else '', al, '\n'.join(take(at_)), ('\n... (%d more lines)' % (len(at_) - budget)) if len(at_) > budget else ''))
        budget -= max(len(bt_), len(at_))
        if budget <= 0: o.append('(remaining changed hunks of this function trimmed)'); break
    o.append('</details>\n')
    return '\n'.join(o)
with open(os.path.join(OUT, '2026-11-27-after-referee-packet.md'), 'w', encoding='utf-8') as f:
    f.write('# Blind referee packet: unmatched rows of the after-count\n\n')
    f.write('Source: active_inference commit cf65a471, tree hash `%s` (branch feat/shared-engine, the end of K3).\n' % TREE_HASH)
    f.write('Rows are the after-count rows whose key, reader or role is new relative to the labelled baseline, rows whose enclosing function changed ("context changed"), and baseline rows that were not found by name or value ("moved or removed?"). No label, no proposed label. The role column quotes the code\'s own comment or the loader text, not a reading.\n\n')
    f.write(pre + '\n')
    f.write("## Referee's rules (header of referee_labels_2026-11-27.tsv, verbatim)\n\n```\n" + '\n'.join(rules) + '\n```\n\n')
    f.write('## Rows to label\n\nColumns: id | key or constant | value | location (file:line at cf65a471) | role (quoted from the code)\n\n')
    for fl in sorted(byfile):
        f.write('\n### %s\n\n' % fl)
        rs = sorted(byfile[fl], key=lineno)
        for a in rs:
            f.write('| %s | %s | %s | %s | %s |\n' % (a['aid'], a['key'].replace('|', '\\|'), str(a['value']).replace('|', '\\|')[:120], a['where'].replace('|', '\\|'), qrole(a).replace('|', '\\|')))
        for a in rs:
            if a.get('ctx'): f.write(ctx_block(a) + '\n')

# stats
st = collections.OrderedDict()
st['baseline rows'] = len(base); st['after rows'] = len(after); st['matched (own)'] = sum(1 for a in matched if not a['inherited'])
st['inherited via reader'] = sum(1 for a in matched if a['inherited']); st['unmatched (packet)'] = len(unmatched)
st['removed'] = len(removed); st['unresolved'] = len(unlocated)
perfile = collections.Counter(a['cd'] for a in matched if a['inherited'])
st['baseline reader fixed from loader'] = READER_FIX
st['present but unread (table countable.toml)'] = len(unread)
st['packet: context changed'] = sum(1 for a in unmatched if a.get('ctx'))
st['packet: moved or removed?'] = sum(1 for a in unmatched if a.get('moved_or_removed'))
st['packet: moved (after-count)'] = sum(1 for a in unmatched if a['note'].startswith('moved (after-count'))
st['packet: loader/consumer changed'] = sum(1 for a in unmatched if 'consumer' in a['note'] or 'reading line' in a['note'])
st['packet: config/hidden other'] = sum(1 for a in unmatched if a['kind'] in ('config','hidden')) - st['packet: moved (after-count)'] - st['packet: loader/consumer changed']
st['packet: src scan/doc rows'] = sum(1 for a in unmatched if a['kind']=='src' and not a.get('ctx') and not a.get('moved_or_removed'))
json.dump(dict(stats=st, inherited_by_class=perfile, unmatched_by_kind=collections.Counter(a['kind'] + ('/scan' if a.get('named') is not None else '') for a in unmatched),
               removed_reasons=collections.Counter(re.sub(r'\(.*', '', v)[:60] for v in removed.values()),
               scope_missing=SCOPE_MISSING, scope_sizes={t: len(s) for t, s in SCOPE.items()},
               unmatched_config_notes=collections.Counter(a['note'][:70] for a in unmatched if a['kind'] in ('config', 'hidden')),
               consumer_notes={'%s/%s'%k:v for k,v in CONSUMER_NOTE.items()}, inherited_disagree=[a['aid'] for a in matched if 'DISAGREE' in a['note']]),
          open(os.path.join(SCR, 'stats.json'), 'w'), indent=1)
print(json.dumps(dict(st), indent=1)); print(dict(perfile))
