"""Extend the after-count map from cf65a471 to the graded-binary tree 17f13198, incrementally (AuRo test c). Labels nothing.

Usage (python3 -I):
  2026-11-27-after-count-17f13198.py MAP_SCRIPT DOC COUNTER_LABELS REFEREE_LABELS REFEREE_PACKET AFTER_REFEREE_LABELS \
      TREE_535d57ec TREE_cf65a471 TREE_17f13198 OUTDIR SCRATCH

It runs the map script twice (TA = cf65a471, TA = 17f13198; the baseline tree and the rules are the same), pairs the rows of the two
runs, and writes, in OUTDIR:
  2026-11-27-after-count-map-17f13198.tsv      every after row at 17f13198; ids of cf65a471 rows are kept when the row is unchanged
                                              (same key, same reader, same role: reading statement and, for a description-only edit,
                                              its consumers; same evidence line and same enclosing function), new ids from A2589 on
  2026-11-27-after-count-removed-17f13198.tsv  baseline ids with no match at 17f13198, and cf65a471 after rows that do not exist at 17f13198
  2026-11-27-after-referee-packet-17f13198.md  the rows that need a label: new, changed since cf65a471, or newly unmatched
The cf65a471 files are not touched.
"""
import sys, os, re, runpy, difflib, collections, subprocess

MAPPY, DOC, CNT, REF, PKT, AREF, T535, TCF, T17, OUT, SCR = sys.argv[1:12]
TREE_HASH_17 = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(DOC)), 'rev-parse', '17f13198^{tree}'], capture_output=True, text=True).stdout.strip() or '5572d3e8d46cebde6a5d52804e30a9e067a3ea7b'
CODE_COMMIT = '17f13198'; BATCH_COMMIT = '91598c88'

def run(ta, tag):
    d = os.path.join(SCR, 'run_' + tag); os.makedirs(d, exist_ok=True)
    sys.argv = [MAPPY, DOC, CNT, REF, PKT, T535, ta, d, d]
    return runpy.run_path(MAPPY, run_name='__map__')
O = run(TCF, 'cf65a471')
N = run(T17, '17f13198')
rd, norm, code_part = N['rd'], N['norm'], N['code_part']

# the labelled pass-2 rows of the cf65a471 packet
P2 = {}
for l in open(AREF, encoding='utf-8'):
    if l.startswith('A'):
        p = l.rstrip('\n').split('\t'); P2[p[0]] = p[1]
oafter, nafter = O['after'], N['after']

def line_text(root, f, ln):
    t = rd(root, f)
    if t is None: return ''
    ls = t.split('\n')
    return norm(code_part(ls[ln - 1])) if 1 <= ln <= len(ls) else ''
def row_line(a):
    m = re.search(r':(\d+)', a['where']); return int(m.group(1)) if m else 0
def ident(a, root):
    k = a['kind']
    if k in ('config', 'hidden'): return ('cfg', a['key'], a['file'], a['reader'], k)
    if a.get('moved_or_removed'): return ('pseudo', a['key'], a['file'])
    if a['mid'] and not a.get('ctx'): return ('rel', a['key'], a['file'], a['mid'])
    if a.get('ctx'): return ('rel', a['key'], a['file'], a['note'].split('baseline ')[-1])
    return ('scan', a['file'], a['key'], line_text(root, a['file'], row_line(a)) if a['file'] else '', a['note'][:12] if 'doc' in a['note'] else '')
# The after-count doc rows carry the cf65a471 file:line of their constant. At 17f13198 that line may have moved: relocate such a row by the
# text of its cf65a471 line; if the text is still in the file and a scanned row or a config row sits there, the doc row is an annotation of it.
FUNC_RELOCATED = []
def relocate_doc_rows_by_function():
    """A doc row carries the cf65a471 line of its constant. If that line was inside a function, the same function at 17f13198
    (same signature) gives the row's line at the same offset (a bare line number can land in another function after edits above it)."""
    for a in nafter:
        if a['kind'] != 'src' or 'after-count row (doc)' not in a['note'] or not a['file']: continue
        old = next((o for o in oafter if o['file'] == a['file'] and o['key'] == a['key']), None)
        if not old: continue
        lo = row_line(old); fb = N['enclosing'](TCF, a['file'], lo)
        if not fb: continue
        cand = [fn for fn in N['functions'](T17, a['file']) if fn['sig'] == fb['sig']]
        if not cand: continue
        fa = min(cand, key=lambda fn: abs(fn['start'] - fb['start']))
        l2 = fa['start'] + (lo - fb['start'])
        if l2 != row_line(a):
            FUNC_RELOCATED.append((a['key'], a['file'], row_line(a), l2))
            a['where'] = '%s:%d' % (a['file'], l2)
relocate_doc_rows_by_function()
STALE = []
def relocate_doc_rows():
    drop = []
    for a in nafter:
        if a['kind'] != 'src' or 'after-count row (doc)' not in a['note'] or not a['file']: continue
        old = next((o for o in oafter if o['kind'] != 'x' and (o['key'] == a['key'] or ('after-count row: ' + a['key']) in o['note']) and o['file'] == a['file']), None) \
              or next((o for o in oafter if o['file'] == a['file'] and row_line(o) == row_line(a) and o is not a), None)
        ln_old = row_line(old) if old else row_line(a)
        txt = line_text(TCF, a['file'], ln_old)
        if not txt: continue
        al = (rd(T17, a['file']) or '').split('\n')
        occ = [i + 1 for i, x in enumerate(al) if norm(code_part(x)) == txt]
        if not occ:
            STALE.append(a); drop.append(a); continue      # the cf65a471 code of this doc row is not in the 17f13198 file
        l2 = min(occ, key=lambda x: abs(x - ln_old))
        hit = next((b for b in nafter if b is not a and b['kind'] == 'src' and b['file'] == a['file'] and row_line(b) == l2), None) or \
              next((b for b in nafter if b['kind'] in ('config', 'hidden') and re.search(re.escape('%s:%d' % (a['file'], l2)) + r'(?!\d)', b['where'])), None)
        if hit:
            hit['note'] = (hit['note'] + '; ' if hit['note'] else '') + 'after-count row: ' + a['key']
            drop.append(a)
    for a in drop: nafter.remove(a)
    return drop
DOC_DROPPED = relocate_doc_rows()
# pair rows of the two runs by identity (in order, for repeated identities)
obuck = collections.defaultdict(list)
for a in oafter: obuck[ident(a, TCF)].append(a)
pair = {}
for a in nafter:
    b = obuck.get(ident(a, T17))
    if b: pair[a['aid']] = b.pop(0)
paired_o = {id(v) for v in pair.values()}

def fn_changed(f, lo, ln):
    """enclosing function of the row, cf65a471 vs 17f13198: None if unchanged, else (before text, after text)."""
    fb = N['enclosing'](TCF, f, lo); fa = N['enclosing'](T17, f, ln)
    if (fb is None) != (fa is None) or (fb and (fb['sig'] != fa['sig'] or fb['body'] != fa['body'])): return fb, fa
    return None
def make_ctx(f, lo, ln, fb, fa, why):
    bt = N['fn_text'](TCF, f, fb) if fb else []; at = N['fn_text'](T17, f, fa) if fa else []
    sm = difflib.SequenceMatcher(None, [norm(x) for x in bt], [norm(x) for x in at], autojunk=False)
    hunks = [(tag, (fb['start'] if fb else 1) + i1, bt[i1:i2], (fa['start'] if fa else 1) + j1, at[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal']
    return dict(sig=(fa or fb)['sig'][:160], hunks=hunks, fn_lines=(len(bt), len(at)), ev_before='%s:%d (cf65a471)' % (f, lo), ev_after='%s:%d (17f13198)' % (f, ln),
                ev_text=line_text(T17, f, ln), why=why)

def cfg_changed(a, o):
    """reading statement (and, for a description-only edit, every consumer) at cf65a471 vs 17f13198."""
    lk = a['loader']
    sb = {x[2] for x in O['LA'][lk].get(a['key'], [])}; sa = {x[2] for x in N['LA'][lk].get(a['key'], [])}
    if sb == sa: return None
    strip = lambda t: re.sub(r'"(?:[^"\\]|\\.)*"', '""', t)
    if {strip(x) for x in sb} != {strip(x) for x in sa}: return 'reading statement changed'
    m = re.match(r'\s*([\w.\[\]>-]+)\s*=\s*[\w:]+\(\s*"', next(iter(sa)))
    if not m: return 'description-only edit, field not identifiable'
    field = re.split(r'[.>-]+', m.group(1))[-1]
    tgt = 'countable_concept' if lk == 'countable' else lk
    files = sorted(N['SCOPE'][tgt])
    if N['consumer_lines'](TCF, files, field) != N['consumer_lines'](T17, files, field): return 'a consumer of %s changed' % field
    return None

# ids and status
next_id = 2589
status = {}   # aid(N run) -> (final id, label source, why)
need = []     # rows for the packet
for a in nafter:
    o = pair.get(a['aid'])
    if a.get('unread'):
        a['fid'] = o['aid'] if o else None; a['fstat'] = 'unread'
        if a['fid'] is None: a['fid'] = 'A%04d' % next_id; next_id += 1
        continue
    why = None
    if a['mid']:
        a['fstat'] = 'matched'            # label = frozen label of the baseline row
    elif o and o['mid']:
        why = 'was matched at cf65a471 (%s), now unmatched' % o['mid']
    elif o is None:
        why = 'new at 17f13198'
    else:
        # unmatched at both: does the pass-2 label of the cf65a471 row still apply?
        assert o['aid'] in P2, o['aid']
        if a['kind'] in ('config', 'hidden'):
            ch = cfg_changed(a, o)
            if ch: why = 'changed since cf65a471: ' + ch
        elif a['file'] and not a.get('moved_or_removed'):
            lo, ln_ = row_line(o), row_line(a)
            r = fn_changed(a['file'], lo, ln_) if lo and ln_ else None
            if r is not None:
                why = 'changed since cf65a471: enclosing function differs'
                a['ctx2'] = make_ctx(a['file'], lo, ln_, r[0], r[1], why)
        if why is None: a['fstat'] = 'pass2'; a['p2label'] = P2[o['aid']]
    if why:
        a['fstat'] = 'packet'; a['why'] = why; a['prev'] = o['aid'] if o else ''
        need.append(a)
    # id
    if o and a['fstat'] in ('matched', 'pass2') and (a['fstat'] == 'pass2' or o['mid'] == a['mid']): a['fid'] = o['aid']
    else: a['fid'] = 'A%04d' % next_id; next_id += 1
# the packet lists new ids in a stable order
need.sort(key=lambda a: int(a['fid'][1:]))
ids_new = [a['fid'] for a in need]
assert len(set(ids_new)) == len(ids_new)

# ---------------------------------------------------------------- outputs
def esc(s): return str(s).replace('\t', ' ').replace('\n', ' ')
def tg(a): return '+'.join(a['tgt']) if a['tgt'] else '(none)'
with open(os.path.join(OUT, '2026-11-27-after-count-map-17f13198.tsv'), 'w', encoding='utf-8') as f:
    f.write('after_row_id\ttarget(s)\tkey\tvalue\twhere\tmatched_baseline_id\tfrozen_label\tinherited_via_reader\tnote\tstatus_vs_cf65a471\tcf65a471_id\n')
    for a in nafter:
        o = pair.get(a['aid'])
        st = {'matched': 'same (baseline label)', 'pass2': 'matched, cf65a471 id %s (pass-2 label stands)%s' % (a.get('fid'), '; A2599 was sent to the referee by mistake (its enclosing function predict_theta_on is unchanged); referee label patch = pass-2 label' if a.get('fid') == 'A2580' else ''), 'unread': 'present but unread', 'packet': 'PACKET: ' + a.get('why', '')}[a['fstat']]
        lab = a['frozen'] if a['fstat'] in ('matched', 'unread') else (a.get('p2label', '') if a['fstat'] == 'pass2' else '')
        f.write('\t'.join(esc(x) for x in [a['fid'], tg(a), a['key'], a['value'], a['where'], a['mid'], lab, a['inherited'], a['note'], st, o['aid'] if o else '']) + '\n')
# removed
base = N['base']; removedN = N['removed']; removedO = O['removed']
unlocN = {r['id'] for r, _ in N['unlocated']}
with open(os.path.join(OUT, '2026-11-27-after-count-removed-17f13198.tsv'), 'w', encoding='utf-8') as f:
    f.write('baseline_id\tsection\tkey\twhere\tbaseline_label\tfrozen_label\twhy\tat_cf65a471\n')
    for r in base:
        if r['id'] in removedN:
            f.write('\t'.join(esc(x) for x in [r['id'], r['sec'], r['name'], r['where'], r['label'], r['frozen'], removedN[r['id']], 'removed too' if r['id'] in removedO else 'NEW at 17f13198 (was matched)']) + '\n')
    for r in base:
        if r['id'] in removedO and r['id'] not in removedN:
            f.write('\t'.join(esc(x) for x in [r['id'], r['sec'], r['name'], r['where'], r['label'], r['frozen'], '(matched at 17f13198)', 'was removed at cf65a471']) + '\n')
    gone = [o for o in oafter if id(o) not in paired_o and not o.get('unread')]
    for o in gone:
        f.write('\t'.join(esc(x) for x in [o['aid'], 'after row of cf65a471', o['key'], o['where'], '', P2.get(o['aid'], o['frozen']), 'after row absent at 17f13198', '']) + '\n')

# packet
pk = open(PKT, encoding='utf-8').read()
pre = pk[pk.index('## Definition'):pk.index('## Rows to label')].rstrip() + '\n'
rules = []
for l in open(REF, encoding='utf-8'):
    if l.startswith('# Rules applied') or (rules and l.startswith('#') and not l.startswith('# Totals')): rules.append(l.rstrip('\n'))
    elif rules: break
rules2 = []
for l in open(AREF, encoding='utf-8'):
    if l.startswith('#') and not l.startswith('# Totals') and not l.startswith('#   '): rules2.append(l.rstrip('\n'))
    elif l.startswith('# Totals'): break
    elif rules2 and l.startswith('#   '): rules2.append(l.rstrip('\n'))
qrole = N['qrole']; ctx_block = N['ctx_block']
def role(a):
    if a.get('ctx'):
        cm = N['code_role'](T17, a['file'], row_line(a))
        return '**context changed** (baseline row %s: the enclosing function differs between the baseline tree 535d57ec and 17f13198; before/after text below). Code comment: %s' % (a['note'].split('baseline ')[-1], ('"' + cm + '"') if cm else '(none)')
    return qrole(a)
def loc_file(a): return a['file'] if a['file'] else '(unlocated)'
byfile = collections.defaultdict(list)
for a in need: byfile[loc_file(a)].append(a)
cnt = collections.Counter()
for a in need:
    w = a['why']
    cnt['new at 17f13198' if w.startswith('new') else 'changed since cf65a471' if w.startswith('changed') else 'was matched, now unmatched'] += 1
    cnt['kind: ' + ('config/hidden' if a['kind'] in ('config', 'hidden') else 'src')] += 1
PKT_OUT = os.path.join(OUT, '2026-11-27-after-referee-packet-17f13198.md')
PKT_FROZEN = os.path.exists(PKT_OUT)   # the packet the referee labelled stays as sent (A2599 was sent by mistake, see Addendum 4b)
with open(PKT_OUT if not PKT_FROZEN else os.path.join(SCR, 'packet_regenerated.md'), 'w', encoding='utf-8') as f:
    f.write('# Blind referee packet 2: rows of the after-count at the graded-binary tree 17f13198\n\n')
    f.write('Source: active_inference commit %s (tree hash `%s`), the code the graded batch was built from; the batch commit %s is docs-only on top of it. The earlier after-count (cf65a471, tree `5c0029ab522691fb4747d2e7215493162f13f0b0`) stays recorded as an intermediate count; its pass-2 labels (A0001-A2588) stand for every row that is unchanged between the two trees. This packet holds only the rows that need a label: new rows, rows whose reading statement, consumers or enclosing function changed since cf65a471, and baseline-matched rows that stopped matching. Ids continue from A2589.\n\n' % (CODE_COMMIT, TREE_HASH_17, BATCH_COMMIT))
    f.write('No label, no proposed label. The role column quotes the code\'s own comment or the loader text, not a reading. "Why in this packet" says which test sent the row here.\n\n')
    f.write(pre + '\n')
    f.write("## Referee's rules\n\nRules 0-8 (header of referee_labels_2026-11-27.tsv, verbatim), then rules 9-13 (header of after_referee_labels_2026-11-27.tsv, verbatim).\n\n```\n" + '\n'.join(rules) + '\n' + '\n'.join(rules2) + '\n```\n\n')
    f.write('## Rows to label\n\nColumns: id | key or constant | value | location (file:line at 17f13198) | why in this packet | role (quoted from the code)\n\n')
    for fl in sorted(byfile):
        f.write('\n### %s\n\n' % fl)
        rs = sorted(byfile[fl], key=lambda x: row_line(x))
        for a in rs:
            r = role(a) if not a.get('ctx2') else '**changed since cf65a471** (the enclosing function differs between cf65a471 and 17f13198; before/after text below). Code comment: ' + ('"' + N['code_role'](T17, a['file'], row_line(a)) + '"' if N['code_role'](T17, a['file'], row_line(a)) else '(none)')
            f.write('| %s | %s | %s | %s | %s | %s |\n' % (a['fid'], a['key'].replace('|', '\\|'), str(a['value']).replace('|', '\\|')[:120], a['where'].replace('|', '\\|'),
                                                           (a['why'] + ((' (was ' + a['prev'] + ')') if a['prev'] else '')).replace('|', '\\|'), r.replace('|', '\\|')))
        for a in rs:
            blk = a.get('ctx2') or a.get('ctx')
            if blk:
                tmp = dict(a); tmp['ctx'] = blk; tmp['aid'] = a['fid']
                f.write(ctx_block(tmp) + '\n')
print('rows at 17f13198:', len(nafter), ' packet:', len(need), dict(cnt), ' next id', next_id)
print('doc rows relocated onto existing rows:', [a['key'] for a in DOC_DROPPED])
print('stale doc rows dropped (code absent at 17f13198):', [a['key'] for a in STALE])
print('doc rows relocated by function:', FUNC_RELOCATED)
print('statuses', collections.Counter(a['fstat'] for a in nafter))
print('removed at 17f13198:', len(removedN), ' at cf65a471:', len(removedO), ' cf65a471 rows gone:', len([o for o in oafter if id(o) not in paired_o and not o.get('unread')]))
