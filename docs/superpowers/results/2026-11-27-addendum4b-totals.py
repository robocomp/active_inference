"""Addendum 4b of the patch-constant baseline: totals at the graded-binary tree 17f13198, all merges applied (AuRo test c). Labels nothing.

Usage (python3 -I):
  2026-11-27-addendum4b-totals.py AFTER17_SCRIPT MAP_SCRIPT DOC COUNTER_LABELS REFEREE_LABELS REFEREE_PACKET AFTER_REFEREE_LABELS \
      AFTER2_REFEREE_LABELS TREE_535d57ec TREE_cf65a471 TREE_17f13198 OUTDIR SCRATCH

It runs the 17f13198 map (which runs the map script twice), reads the three frozen label sources (baseline + pass 1, pass 2, pass 3),
charges rows to classes by the graded-batch binary (as Addendum 4), applies the merges of rule (i) on both sides, and writes the
"## Addendum 4b" section of DOC (after Addendum 4, before "## After part 1"). Nothing else in DOC changes.
"""
import sys, os, re, runpy, hashlib, collections

A17, MAPPY, DOC, CNT, REF, PKT, AREF, AREF2, T535, TCF, T17, OUT, SCR = sys.argv[1:14]
sys.argv = [A17, MAPPY, DOC, CNT, REF, PKT, AREF, T535, TCF, T17, OUT, SCR]
G = runpy.run_path(A17, run_name='__a17__')
N = G['N']; nafter = G['nafter']; P2 = G['P2']
base, SCOPE, TF = N['base'], N['SCOPE'], N['targets_files']
baseid = {r['id']: r for r in base}

P3 = {}
for l in open(AREF2, encoding='utf-8'):
    if l.startswith('A'):
        p = l.rstrip('\n').split('\t'); P3[p[0]] = p[1]
assert len(P3) == 11 and sorted(P3) == ['A%04d' % k for k in range(2589, 2600)], sorted(P3)
CATS = ['class', 'sensor', 'patch', 'switch (patch)', 'excluded', 'switch (other)']
def cat(x):
    x = x.rstrip('?'); assert x in CATS, x; return x
def is_patch(c): return c in ('patch', 'switch (patch)')
def count(labels):
    c = collections.Counter(labels); return [c[k] for k in CATS]

# ---- A2599 (sent by mistake): it is A2580, pass-2 label stands
a2580 = next(a for a in nafter if a['fid'] == 'A2580')
assert a2580['fstat'] == 'pass2'
A2599_AGREES = P3['A2599'] == P2['A2580']
ids_packet = sorted(a['fid'] for a in nafter if a['fstat'] == 'packet')
assert ids_packet == ['A%04d' % k for k in range(2589, 2599)], ids_packet

CLASSES = ['chair', 'table', 'refrigerator', 'hood', 'nightstand', 'bed', 'toilet']
CLS_DIR = {c: c + '_concept' for c in CLASSES}
CLS_T = {'chair': 'chair_concept', 'table': 'table_concept', 'refrigerator': 'refrigerator_concept', 'hood': 'hood_concept',
         'nightstand': 'countable_concept', 'bed': 'countable_concept', 'toilet': 'countable_concept'}
# ---------------------------------------------------------------- BEFORE (addendum 2 build rule, as in Addendum 4)
def owner(r):
    s = r['sec']
    for k in CLASSES:
        if s.startswith(k + ' ') or s.startswith(k + ' (') or s == k: return ('own', k)
    if 'Own-directory' in s: return ('own', r['cls'])
    if s.startswith('countable engine'): return ('engine', None)
    return ('common', None)
def files_of(r): return re.findall(r'(common/[\w/.\-]+\.(?:h|cpp|hpp))', r['where'])
def charged_before(r, c):
    kind, o = owner(r)
    if kind == 'own': return o == c
    if kind == 'engine': return c in ('nightstand', 'bed', 'toilet')
    fs = files_of(r)
    return bool(fs) and any(f in TF[CLS_T[c]] for f in fs)
def blabel(r, v):
    if v == 'r057' and r['id'] == 'R057': return 'excluded'
    return cat(r['frozen'])
# ---------------------------------------------------------------- AFTER (graded binaries, tree 17f13198)
AFTER_TOML = {'chair': 'chair_concept/etc/countable.toml', 'table': 'table_concept/etc/config.toml',
              'refrigerator': 'refrigerator_concept/etc/countable.toml', 'hood': 'hood_concept/etc/countable.toml',
              'nightstand': 'nightstand_concept/etc/config.toml', 'bed': 'bed_concept/etc/config.toml', 'toilet': 'toilet_concept/etc/config.toml'}
AFTER_BUILD = {c: ('table_concept' if c == 'table' else 'countable_concept') for c in CLASSES}
def charged_after(a, c):
    if a['fstat'] == 'unread': return False
    if a['kind'] in ('config', 'hidden'): return a['file'] == AFTER_TOML[c]
    return bool(a['file']) and a['file'] in SCOPE[AFTER_BUILD[c]]
def alabel(a):
    if a['fstat'] == 'pass2': return cat(P2[a['fid']])
    if a['fstat'] == 'packet': return cat(P3[a['fid']])
    assert a['fstat'] == 'matched', a['fstat']
    if a['mid'] == 'R057': return 'excluded'
    return cat(a['frozen'])
for a in nafter:
    if a['fstat'] != 'unread': a['lab'] = alabel(a)

# ---------------------------------------------------------------- merges
LA = N['LA']; LB = N['LB']
FIELD_RE = re.compile(r'\b(\w+)\s*=\s*[\w:]+\(\s*(?:\w+\s*,\s*)?"([A-Z]\w*(?:\.\w+)+)"')
def field_key(loader):     # cfg field name -> toml key, from the loader source (getf("K", ...) and manifest::resolve(cfg, "K", ...))
    d = {}
    t = N['rd'](T17, N['LOADER'][loader]) or ''
    for m in FIELD_RE.finditer(t): d.setdefault(m.group(1), m.group(2))
    return d
FK = {lk: field_key(lk) for lk in LA}
BEFORE_LOADER = {'chair': 'chair_concept', 'table': 'table_concept', 'refrigerator': 'refrigerator_concept', 'hood': 'hood_concept',
                 'nightstand': 'countable', 'bed': 'countable', 'toilet': 'countable'}
AFTER_LOADER = {c: ('table_concept' if c == 'table' else 'countable') for c in CLASSES}

def ids_named(name, file_part=None):
    return [r['id'] for r in base if r['name'] == name and (file_part is None or file_part in r['where'])]
# triple families: library rows (baseline ids) that a class's config sets from a toml key. field: per loader family (None = not set from config
# in that build). Evidence (file:line in 17f13198): the assignment of the cfg field to the library member.
def fam(name, rows, fields, evid): return dict(name=name, rows=rows, fields=fields, evid=evid)
ALL5 = lambda f: {k: f for k in ('chair_concept', 'table_concept', 'refrigerator_concept', 'hood_concept', 'countable')}
NOCHAIR = lambda f: {'chair_concept': None, 'table_concept': f, 'refrigerator_concept': f, 'hood_concept': f, 'countable': f}
TRIPLES = [
    fam('CarveParams.sensor_sigma_m', ['B1160'], NOCHAIR('existence_sensor_sigma_m'), 'countable_existence.cpp:63, table_existence.cpp:59, refrigerator_existence.cpp:60, hood_existence.cpp:60'),
    fam('CarveParams.detection_prob', ['B1161'], NOCHAIR('existence_detection_prob'), 'countable_existence.cpp:64, table_existence.cpp:60 (and the same in refrigerator, hood)'),
    fam('CarveParams.clutter_prob', ['R281'], NOCHAIR('existence_clutter_prob'), 'countable_existence.cpp:65, table_existence.cpp:61 (and the same in refrigerator, hood)'),
    fam('ExistenceBelief L_max default + RemovalPolicy.logodds_max', ['B1162', 'B1165'], dict(NOCHAIR('existence_logodds_max'), chair_concept='exist_max_logodds'),
        'countable_existence.cpp:79, specificworker_lifecycle.cpp:584-585, :602, specificworker_consumers.cpp:177 (set_max); chair_existence.cpp:66'),
    fam('RemovalPolicy.removal_prob', ['B1166'], NOCHAIR('existence_removal_prob'), 'countable_existence.cpp:80, table_existence.cpp:76 (chair derives it from exist_remove_logodds: not a copy, not merged)'),
    fam('RemovalPolicy.frame_correlation', ['B1167'], NOCHAIR('existence_frame_correlation'), 'countable_existence.cpp:81, table_existence.cpp:77'),
    fam('RemovalPolicy.remove_frames', ['B1168'], dict(NOCHAIR('existence_remove_frames'), chair_concept='exist_remove_frames'), 'countable_existence.cpp:82, chair_existence.cpp:68'),
    fam('TrackerParams.gate_mahalanobis', ['B1179'], ALL5('tracker_gate_mahalanobis'), 'countable specificworker_lifecycle.cpp:156, chair specificworker.cpp:508'),
    fam('TrackerParams.gate_fallback_m', ['B1180'], ALL5('tracker_gate_fallback_m'), 'countable specificworker_lifecycle.cpp:157, chair specificworker.cpp:509'),
    fam('TrackerParams.detection_noise_m', ['B1181'], ALL5('tracker_detection_noise_m'), 'countable specificworker_lifecycle.cpp:158, chair specificworker.cpp:510'),
    fam('TrackerParams.birth_frames', ['B1182'], ALL5('tracker_birth_frames'), 'countable specificworker_lifecycle.cpp:159, chair specificworker.cpp:511'),
    fam('TrackerParams.birth_min_sep_m', ['B1183'], ALL5('tracker_birth_min_sep_m'), 'countable specificworker_lifecycle.cpp:161, chair specificworker.cpp:520'),
    fam('support_bank height margin (library radius/height row, library below_m, class "support select below_m" rows)',
        ['B1210', 'B1211', 'B0123', 'B0274', 'B0451', 'B1148'] + ids_named('support select below_m', 'hood_concept'), ALL5('support_select_height_margin_m'),
        '<class>_support_bank.h:31-32 (prm.height_margin_m = cfg.support_select_height_margin_m; below_m from the same field), countable_support_bank.h:32'),
    fam('support_bank.quantization_m', ['B1212'], ALL5('support_bank_quantization_m'), '<class>_support_bank.h (prm quantization from cfg.support_bank_quantization_m)'),
    fam('DetectorEnvelope.min_fill', ['B1171'], ALL5('detect_min_fill'), 'countable_card.h:34 (sp.env = {cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft}) and the other cards'),
    fam('DetectorEnvelope.max_fill', ['B1172'], ALL5('detect_max_fill'), 'same init list'),
    fam('DetectorEnvelope.soft', ['B1173'], ALL5('detect_soft'), 'same init list'),
    fam('Detectability.periph_ref', ['B1175'], {'chair_concept': 'ai2_periph_ref', 'table_concept': None, 'refrigerator_concept': 'ai2_periph_ref', 'hood_concept': 'ai2_periph_ref', 'countable': 'ai2_periph_ref'},
        'countable specificworker_lifecycle.cpp:230, chair specificworker.cpp:566 (table builds the reference as a literal: not merged)'),
    fam('Detectability.range_ref_m', ['B1176'], {'chair_concept': None, 'table_concept': 'existence_absence_range_ref_m', 'refrigerator_concept': 'existence_absence_range_ref_m', 'hood_concept': 'existence_absence_range_ref_m', 'countable': 'existence_absence_range_ref_m'},
        'countable specificworker_lifecycle.cpp:231, table specificworker_lifecycle.cpp:143 (chair uses the literal 2.5f: not merged)'),
    fam('lidar_select Weights.coverage_n0 default', ['B1194'], NOCHAIR('lidar_coverage_n0'), 'countable_lidar_range_channel.cpp:79, :172'),
    fam('lidar_select Weights.robust_c_m default', ['B1195'], NOCHAIR('lidar_robust_c_m'), 'countable_lidar_range_channel.cpp:79, :172'),
    fam('BeamRays.sigma_s default + beam staging sigma_s default (the composite row also holds rho_max, which stays its own constant)', ['R279', 'B1140'],
        {'chair_concept': None, 'table_concept': None, 'refrigerator_concept': None, 'hood_concept': None, 'countable': 'model_sigma_m'},
        'countable_lidar_range_channel.cpp:174, countable_lidar_range_channel.h:78 (st.sigma_s / L.sigma_s = cfg_.model_sigma_m)'),
]
# same-quantity merges (two literal sites of one parameter, the code shows it): (name, kept, [dropped]); before side in baseline ids,
# after side in selectors ('Am', baseline id of the matched after row) or ('A', after id)
SAME_B = [('chair kFlipClamp, one named constant used twice (chair_belief.cpp:257, :274)', 'B0114', ['B0116']),
          ('refrigerator front clamp: kFrontClamp 6.0f and the +-6.0f re-baseline literal (refrigerator_belief.cpp:694, :668)', 'B0442', ['B0443']),
          ('hood front clamp (hood_belief.cpp:753, :727)', 'B0612', ['B0613']),
          ('countable front clamp (countable_belief.cpp:1717, :1549)', 'B1134', ['B1135']),
          ('contour_edge control offsets (contour_edge_project.h:95, contour_edge_check.cpp:86)', 'R290', ['R291']),
          ('log-odds cap literals (existence_belief.h:483, :754, chair_instance.h:118)', 'B1162', ['B1165', 'R022'])]
SAME_A = [('3-sigma z-band edges (countable_fitter.cpp:1180, :1181)', ('A', 'A2288'), [('A', 'A2550')]),
          ('countable front clamp (countable_belief.cpp:1380 kFrontClamp, :1221 re-baseline literal)', ('A', 'A2268'), [('A', 'A2269')]),
          ('volatility memory 0.995 (countable_belief.cpp:268 kLambda, persistent_engine.h:116)', ('Am', 'R256'), [('A', 'A2464')]),
          ('log-odds cap literals (existence_belief.h:483, :754 and specificworker.h:250)', ('Am', 'B1162'), [('Am', 'B1165'), ('A', 'A2565')]),
          ('tol_sigma and gn_tol_sigma, one parameter (persistent_engine.h:98, countable_belief.h:332)', ('A', 'A2579'), [('A', 'A2463')])]
LINK_DROP = ['A2537', 'A2538', 'A2539', 'A2540', 'A2594', 'A2595', 'A2596', 'A2597']   # struct default and loader literals of Link*/GhostEvidenceGain
LINK_KEYS = {'A2537': 'Tracker.LinkPiMove', 'A2594': 'Tracker.LinkPiMove', 'A2538': 'Tracker.LinkPiGone', 'A2595': 'Tracker.LinkPiGone',
             'A2539': 'Tracker.LinkLambdaNew', 'A2596': 'Tracker.LinkLambdaNew', 'A2540': 'Tracker.GhostEvidenceGain', 'A2597': 'Tracker.GhostEvidenceGain'}
fid_row = {a['fid']: a for a in nafter}
for k in LINK_DROP + ['A2288', 'A2550', 'A2268', 'A2269', 'A2464', 'A2565', 'A2579', 'A2463']: assert k in fid_row, k

def items_before(c, v):
    return [dict(uid=r['id'], label=blabel(r, v), row=r, name=r['name']) for r in base if charged_before(r, c)]
def items_after(c):
    return [dict(uid=a['fid'], label=a['lab'], row=a, name=a['key']) for a in nafter if a['fstat'] != 'unread' and charged_after(a, c)]
def key_row_item(items, c, side, key):
    for it in items:
        r = it['row']
        if side == 'B':
            if r['kind'] in ('config', 'hidden') and r['name'] == key and r['home'] == CLS_DIR[c]: return it
        else:
            if r['kind'] in ('config', 'hidden') and r['key'] == key: return it
    return None
def find(items, side, sel):
    k, v = sel
    for it in items:
        r = it['row']
        if side == 'B' and k == 'B' and it['uid'] == v: return it
        if side == 'A' and k == 'A' and it['uid'] == v: return it
        if side == 'A' and k == 'Am' and r['mid'] == v and not r['inherited']: return it
    return None

MERGE_LOG = {'B': collections.defaultdict(list), 'A': collections.defaultdict(list)}   # side -> kind -> records
CONFLICTS = []
def apply_merges(items, c, side):
    """returns (kept items, dropped items); logs every merge."""
    parent = {it['uid']: it['uid'] for it in items}
    byuid = {it['uid']: it for it in items}
    def fnd(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    prefer = {}   # component root -> preferred kept uid, set by triples first
    specs = []
    loader = BEFORE_LOADER[c] if side == 'B' else AFTER_LOADER[c]
    build = CLS_T[c] if side == 'B' else AFTER_BUILD[c]
    bl = {'chair_concept': 'chair_concept', 'table_concept': 'table_concept', 'refrigerator_concept': 'refrigerator_concept', 'hood_concept': 'hood_concept', 'countable_concept': 'countable'}[build]
    for f in TRIPLES:
        field = f['fields'].get(bl)
        if not field: continue
        key = FK[bl].get(field)
        if not key: continue
        ki = key_row_item(items, c, side, key)
        if ki is None: continue
        mem = [find(items, side, ('B' if side == 'B' else 'Am', rid)) for rid in f['rows']]
        mem = [m for m in mem if m]
        if mem: specs.append(('triple', f['name'], ki, mem, key))
    for nm, kept, dropped in (SAME_B if side == 'B' else []):
        k = find(items, side, ('B', kept)); d = [find(items, side, ('B', x)) for x in dropped]; d = [x for x in d if x]
        if k and d: specs.append(('same', nm, k, d, ''))
        elif d and not k: specs.append(('same', nm, d[0], d[1:], ''))
    for nm, kept, dropped in (SAME_A if side == 'A' else []):
        k = find(items, side, kept); d = [find(items, side, x) for x in dropped]; d = [x for x in d if x]
        if k and d: specs.append(('same', nm, k, d, ''))
        elif d and not k: specs.append(('same', nm, d[0], d[1:], ''))
    if side == 'A':
        for aid in LINK_DROP:
            d = byuid.get(aid)
            if not d: continue
            ki = key_row_item(items, c, 'A', LINK_KEYS[aid])
            if ki: specs.append(('link', 'Link*/GhostEvidenceGain: the toml key, the loader default literal and the struct default are one constant', ki, [d], LINK_KEYS[aid]))
    for typ, nm, kept, dropped, key in specs:
        for d in dropped:
            if d is kept: continue
            parent[fnd(d['uid'])] = fnd(kept['uid'])
    cand = collections.defaultdict(list)
    for typ, nm, kept, dropped, key in specs:
        cand[fnd(kept['uid'])].append((0 if typ in ('triple', 'link') else 1, kept['uid']))
    for r, lst in cand.items(): prefer[r] = sorted(lst)[0][1]
    comps = collections.defaultdict(list)
    for it in items: comps[fnd(it['uid'])].append(it)
    kept_items, dropped_items = [], []
    for r, mem in comps.items():
        if len(mem) == 1: kept_items.append(mem[0]); continue
        kuid = prefer.get(r, mem[0]['uid']); kept = byuid[kuid]
        kept_items.append(kept)
        for m in mem:
            if m is kept: continue
            dropped_items.append(m)
            if m['label'] != kept['label']: CONFLICTS.append((side, c, m['uid'], m['name'], m['label'], kept['uid'], kept['name'], kept['label']))
    # log
    for typ, nm, kept, dropped, key in specs:
        MERGE_LOG[side][typ].append((c, nm, kept['uid'], kept['name'], [d['uid'] for d in dropped if d is not kept], key))
    return kept_items, dropped_items

RES = {}
for c in CLASSES:
    for v in ('frozen', 'r057'):
        ib = items_before(c, v); kb, db = apply_merges(ib, c, 'B')
        RES[('B', v, c)] = dict(items=kb, dropped=db, raw=ib)
    ia = items_after(c); ka, da = apply_merges(ia, c, 'A')
    RES[('A', 'x', c)] = dict(items=ka, dropped=da, raw=ia)
# the merge log is filled twice for 'B' (two versions): keep one
def dedup_log(side):
    seen = set(); out = []
    for rec in MERGE_LOG[side]['same'] + MERGE_LOG[side]['triple'] + MERGE_LOG[side]['link']:
        k = (rec[0], rec[1], rec[2], tuple(rec[4]))
        if k in seen: continue
        seen.add(k); out.append(rec)
    return out
CONFLICTS = list(dict.fromkeys(CONFLICTS))

# ---------------------------------------------------------------- totals and movement
TOT = {}
for c in CLASSES:
    for v in ('frozen', 'r057'): TOT[('B', v, c)] = count([it['label'] for it in RES[('B', v, c)]['items']])
    TOT[('A', 'x', c)] = count([it['label'] for it in RES[('A', 'x', c)]['items']])
    TOT[('Braw', 'frozen', c)] = count([it['label'] for it in RES[('B', 'frozen', c)]['raw']])
    TOT[('Araw', 'x', c)] = count([it['label'] for it in RES[('A', 'x', c)]['raw']])
any_after = {a['mid'] for a in nafter if a['mid'] and a['fstat'] != 'unread'}
MOVE = {}
for c in CLASSES:
    ia = RES[('A', 'x', c)]['items']
    own = [it for it in ia if it['row']['mid'] and not it['row']['inherited']]
    for v in ('frozen', 'r057'):
        ib = RES[('B', v, c)]['items']
        bp = {it['uid'] for it in ib if is_patch(it['label'])}
        ball = {it['uid'] for it in ib}
        retained = {it['row']['mid'] for it in own}
        removed = sorted(i for i in bp if i not in retained)
        gone = [i for i in removed if i not in any_after]
        outs = [i for i in removed if i in any_after]
        scope_in = [it for it in own if it['row']['mid'] not in ball and is_patch(it['label'])]
        relabel = [it for it in own if it['row']['mid'] in ball and is_patch(blabel(baseid[it['row']['mid']], v)) != is_patch(it['label'])]
        rn = sum(1 for it in relabel if is_patch(it['label'])) - sum(1 for it in relabel if not is_patch(it['label']))
        inh = [it for it in ia if it['row']['inherited']]
        inh_p = [it for it in inh if is_patch(it['label'])]
        added = [it for it in ia if it['row']['fstat'] in ('pass2', 'packet') and is_patch(it['label'])]
        after_p = sum(1 for it in ia if is_patch(it['label']))
        recon = len(bp) - len(removed) + len(scope_in) + len(inh_p) + len(added) + rn
        MOVE[(v, c)] = dict(before=len(bp), removed=len(removed), gone=len(gone), outs=len(outs), scope_in=len(scope_in), inh_all=len(inh), inh_p=len(inh_p), added=len(added),
                            relabel=rn, after=after_p, recon=recon, gone_ids=gone, added_items=added)

# ---------------------------------------------------------------- render
def row6(v): return ' | '.join(str(x) for x in v)
def pt(v): return v[2] + v[3]
sha3 = hashlib.sha256(open(AREF2, 'rb').read()).hexdigest()
sha2 = hashlib.sha256(open(AREF, 'rb').read()).hexdigest()
o = []; w = o.append
w('## Addendum 4b: totals at 17f13198, all merges (2026-11-27)\n')
w('Addendum 4 above is the intermediate count at `cf65a471` and is left as it was. This section is the count at the registered tree of the graded binaries, `17f13198` (tree `5572d3e8d46cebde6a5d52804e30a9e067a3ea7b`; `91598c88` is docs-only on top), with every merge of rule (i) applied on both sides. Mechanical: `2026-11-27-addendum4b-totals.py` (runs `2026-11-27-after-count-17f13198.py`, which runs the map script).\n')
w('- Labels, all frozen and read-only in the AuRo registry: the baseline labels and referee pass 1 (`referee_labels_2026-11-27.tsv`); referee pass 2 (`after_referee_labels_2026-11-27.tsv`, sha256 `%s`, 286 rows A0001-A2588); referee pass 3 (`after2_referee_labels_2026-11-27.tsv`, sha256 `%s`, 11 rows A2589-A2599, no new rule). Rules 9-13 apply to both counts; the only first-pass row they change is R057 (excluded), shown as the second baseline version.' % (sha2, sha3))
w('- Binaries (Addendum 4, rule iii, same evidence): table = the legacy `table_concept` build; chair, refrigerator, hood = `countable_concept` plus the class\'s own `etc/countable.toml`; nightstand, bed, toilet = `countable_concept` plus their own `etc/config.toml`. The graded chair runs `countable_concept` (`2026-11-26-part1-grading.md:220-222`), so A2584 (`chair_belief.cpp:257`) is charged to the chair before only. BEFORE = the addendum 2 build rule.')
w('- File lists at `17f13198` are those of `cf65a471` (no `#include` line, CMake file or source file changed between them; see the end of Addendum 4), so the build scope is the same.')
w('')
w('### A2599 was sent to the referee by mistake')
w('')
w('A2599 (the Sigma cap, `persistent_engine.h`) was placed at the after-count doc\'s line 1514, which at `17f13198` is inside `learn_volatility`; its real function is `predict_theta_on` (now `persistent_engine.h:1551-1560`, code identical to `cf65a471`). The doc row was relocated by function and offset (the script now does this for every doc row): it is the row A2580 with its pass-2 label, status "matched, cf65a471 id A2580" in `2026-11-27-after-count-map-17f13198.tsv`, counted once. The referee labelled A2599 %s; the pass-2 label of A2580 is %s: %s. The packet `2026-11-27-after-referee-packet-17f13198.md` stays as it was sent (11 rows); the map has 10 rows from it (A2589-A2598).' % (P3['A2599'], P2['A2580'], 'they agree' if A2599_AGREES else 'THEY DISAGREE'))
w('')
w('### Totals per class')
w('')
w('Merged on both sides. "before, unmerged" is the Addendum 4 figure for comparison (cf65a471 build rule, no merge).\n')
w('| class | version | class | sensor | patch | switches (patch) | excluded | switches (other) | patch total | total rows |')
w('|---|---|---|---|---|---|---|---|---|---|')
for c in CLASSES:
    for key, nm in ((('Braw', 'frozen', c), 'before, unmerged (frozen)'), (('B', 'frozen', c), 'before, frozen, merged'), (('B', 'r057', c), 'before, R057 excluded, merged'), (('Araw', 'x', c), 'after, unmerged'), (('A', 'x', c), 'after, merged')):
        vv = TOT[key]; w('| %s | %s | %s | %d | %d |' % (c, nm, row6(vv), pt(vv), sum(vv)))
w('')
w('### Movement of the patch total (merged sets)\n')
w('Definitions as in Addendum 4: removed = baseline patch rows with no after row of the class (gone) or outside its new binary (out of scope); scope in = baseline patch rows that now count and did not; inherited = rows of the class\'s own `countable.toml` that took their label through the reader (all in brackets); added = patch rows labelled by referee pass 2 or 3 and charged to the class; relabelled = retained rows whose patch status changes (R057). All on the merged sets, so a merged-away row appears in none of the columns.\n')
w('| class | version | before | removed (gone + out of scope) | scope in | inherited (all) | added | relabelled | reconstructed after | after | gap |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for c in CLASSES:
    for v in ('frozen', 'r057'):
        r = MOVE[(v, c)]
        w('| %s | %s | %d | -%d (%d + %d) | +%d | +%d (%d) | +%d | %+d | %d | %d | %+d |' % (c, 'frozen' if v == 'frozen' else 'R057 excluded', r['before'], r['removed'], r['gone'], r['outs'], r['scope_in'], r['inh_p'], r['inh_all'], r['added'], r['relabel'], r['recon'], r['after'], r['after'] - r['recon']))
gaps = [(c, v, MOVE[(v, c)]['after'] - MOVE[(v, c)]['recon']) for c in CLASSES for v in ('frozen', 'r057') if MOVE[(v, c)]['after'] != MOVE[(v, c)]['recon']]
w('')
w('Reconciliation gaps (shown, not forced): ' + ('; '.join('%s %s %+d' % g for g in gaps) if gaps else 'none, the gap is 0 in every class and both versions') + '.')
w('')
w('Merged away per class (rows dropped from the count, patch ones in brackets), before | after:\n')
w('| class | before (frozen) | after |')
w('|---|---|---|')
for c in CLASSES:
    db = RES[('B', 'frozen', c)]['dropped']; da = RES[('A', 'x', c)]['dropped']
    w('| %s | %d (%d) | %d (%d) |' % (c, len(db), sum(1 for x in db if is_patch(x['label'])), len(da), sum(1 for x in da if is_patch(x['label']))))
w('')
w('### Merges, same quantity at two literal sites (auro rulings)\n')
w('Before side (baseline ids):')
for nm, kept, dropped in SAME_B: w('- %s: %s into %s.' % (nm, ', '.join(dropped), kept))
w('')
w('After side:')
for nm, kept, dropped in SAME_A: w('- %s: %s into %s.' % (nm, ', '.join(d[1] for d in dropped), kept[1] + (' (the after row matched to that baseline row)' if kept[0] == 'Am' else '')))
w('- Link*/GhostEvidenceGain: A2537-A2540 (struct defaults, `countable_config.h:454-457`) and A2594-A2597 (loader literals, `countable_config.cpp:431-434`, which replace A2533-A2536 of the cf65a471 map) into the per-class toml/hidden-default rows of `Tracker.LinkPiMove`, `LinkPiGone`, `LinkLambdaNew`, `GhostEvidenceGain`.')
w('')
w('### Merges, toml key / loader default literal / library struct default of one constant (new in this section)\n')
w('A library struct default that a class\'s loader overwrites from a toml key is one constant with that key and the key\'s loader default (rule i); the merged row takes the key row\'s label. A library row merges for a class only if that class\'s binary shows the assignment. Evidence is the assignment in the 17f13198 sources.\n')
w('| family (library rows) | baseline ids | key (countable loader) | evidence |')
w('|---|---|---|---|')
for f in TRIPLES:
    key = FK['countable'].get(f['fields'].get('countable') or '', '-')
    w('| %s | %s | %s | %s |' % (f['name'], ', '.join(f['rows']), key, f['evid']))
w('')
w('Triples merged, by side and class (library rows dropped into the key row):\n')
for side, nm in (('B', 'Before'), ('A', 'After')):
    w('**%s side**' % nm)
    cnt = collections.defaultdict(list)
    for rec in dedup_log(side):
        if rec[1] and rec[1] in [f['name'] for f in TRIPLES]: cnt[rec[0]].append(rec)
    for c in CLASSES:
        recs = cnt.get(c, [])
        w('- %s: %d families, %d library rows: %s' % (c, len(recs), sum(len(r[4]) + (1 if r[2] and r[2] not in r[4] and False else 0) for r in recs),
                                                    '; '.join('%s [%s] into %s' % (r[1].split(' (')[0], ', '.join(r[4]), r[5]) for r in recs) or '-'))
    w('')
w('Examined and not merged: `angular_margin_rad` (`peripheral_channel.h`, set from two different keys in two paths), `BeamRays.rho_max` and the `rho_max` half of the beam staging row (set from the ray factor\'s `max_range_m`, no toml key), `B1137`/`B1138` (`class_p_label_true`, `headboard_t_m`: overridden from the class manifest, which is not a counted config), `RemovalPolicy.removal_prob` for the chair (derived from `exist_remove_logodds`), table\'s birth `periph_ref` and the chair\'s birth `range_ref_m` (literals in those binaries).')
w('')
w('### Label conflicts (kept row = the key row, or the first-named row for same-quantity merges)\n')
if CONFLICTS:
    w('| side | class | dropped row | its label | kept row | its label |')
    w('|---|---|---|---|---|---|')
    for (side, c, du, dn, dl, ku, kn, kl) in CONFLICTS:
        w('| %s | %s | %s %s | %s | %s %s | %s |' % ('before' if side == 'B' else 'after', c, du, dn[:40], dl, ku, kn[:40], kl))
else:
    w('None.')
w('')
sec = '\n'.join(o) + '\n\n'
s = open(DOC, encoding='utf-8').read()
a = s.index('## Addendum 4b:') if '## Addendum 4b:' in s else None
b = s.index('## After part 1')
s = (s[:a] if a is not None else s[:b]) + sec + '---\n\n' + s[b:]
open(DOC, 'w', encoding='utf-8').write(s)
print('gaps:', gaps); print('conflicts:', len(CONFLICTS))
for c in CLASSES:
    print(c, 'before raw', TOT[('Braw', 'frozen', c)], 'B merged', pt(TOT[('B', 'frozen', c)]), pt(TOT[('B', 'r057', c)]), 'after', pt(TOT[('A', 'x', c)]), 'raw', pt(TOT[('Araw', 'x', c)]), {k: MOVE[('frozen', c)][k] for k in ('before', 'removed', 'scope_in', 'inh_p', 'inh_all', 'added', 'relabel', 'after', 'recon')})
