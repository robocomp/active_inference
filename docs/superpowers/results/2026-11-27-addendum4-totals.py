"""Addendum 4 of the patch-constant baseline: before/after totals per class, mechanical (AuRo test c). Labels nothing.

Usage (python3 -I):
  2026-11-27-addendum4-totals.py MAP_SCRIPT DOC COUNTER_LABELS REFEREE_LABELS REFEREE_PACKET AFTER_REFEREE_LABELS TREE_BEFORE TREE_AFTER OUTDIR SCRATCH
It runs the row-map script (same arguments, so every row, id, scope and match comes from it), reads the frozen pass-2 labels, charges
rows to classes, and rewrites the "## Addendum 4" section of DOC (between Addendum 3 and "## After part 1"). Nothing else in DOC changes.
"""
import sys, os, re, runpy, hashlib, collections

MAPPY, DOC, CNT, REF, PKT, AREF, TB, TA, OUT, SCR = sys.argv[1:11]
sys.argv = [MAPPY, DOC, CNT, REF, PKT, TB, TA, OUT, SCR]
g = runpy.run_path(MAPPY, run_name='__map__')
base, after, SCOPE, TF = g['base'], g['after'], g['SCOPE'], g['targets_files']

# ---------------------------------------------------------------- labels
P2 = {}
for l in open(AREF, encoding='utf-8'):
    if l.startswith('A'):
        p = l.rstrip('\n').split('\t'); P2[p[0]] = (p[1], p[3])
assert len(P2) == 286, len(P2)
CATS = ['class', 'sensor', 'patch', 'switch (patch)', 'excluded', 'switch (other)']
def cat(x):
    x = x.rstrip('?')
    assert x in CATS, x
    return x
def is_patch(c): return c in ('patch', 'switch (patch)')
R057 = next(r for r in base if r['id'] == 'R057')
assert R057['name'] == 'kEqualVolumeDilation'

CLASSES = ['chair', 'table', 'refrigerator', 'hood', 'nightstand', 'bed', 'toilet']
CLS_T = {'chair': 'chair_concept', 'table': 'table_concept', 'refrigerator': 'refrigerator_concept', 'hood': 'hood_concept',
         'nightstand': 'countable_concept', 'bed': 'countable_concept', 'toilet': 'countable_concept'}
# ---------------------------------------------------------------- BEFORE: addendum 3's charging rule, applied to every row
def owner(r):
    s = r['sec']
    for k in ('chair', 'table', 'refrigerator', 'hood', 'nightstand', 'bed', 'toilet'):
        if s.startswith(k + ' ') or s.startswith(k + ' (') or s == k: return ('own', k)
    if 'Own-directory' in s: return ('own', r['cls'])
    if s.startswith('countable engine'): return ('engine', None)
    return ('common', None)
def files_of(r): return re.findall(r'(common/[\w/.\-]+\.(?:h|cpp|hpp))', r['where'])
def charged_before(r, c):
    kind, o = owner(r)
    if kind == 'own': return o == c
    if kind == 'engine': return CLS_T[c] == 'countable_concept' and c in ('nightstand', 'bed', 'toilet')
    fs = files_of(r)
    return bool(fs) and any(f in TF[CLS_T[c]] for f in fs)
def blabel(r, v):   # v: 'frozen' or 'r057'
    if v == 'r057' and r['id'] == 'R057': return 'excluded'
    return cat(r['frozen'])
def count(cats):
    c = collections.Counter(cats); return [c[k] for k in CATS]
before = {}
for v in ('frozen', 'r057'):
    for c in CLASSES:
        before[(v, c)] = count([blabel(r, v) for r in base if charged_before(r, c)])
# check against addendum 3 as printed in DOC
txt = open(DOC, encoding='utf-8').read()
a3 = txt[txt.index('### Combined totals per class (build rule, referee labels'):txt.index('### Counting rules for the after-count')]
for line in a3.split('\n'):
    m = re.match(r'\| (chair|table|refrigerator|hood|nightstand|bed|toilet) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|', line)
    if m:
        c = m.group(1); doc = [int(m.group(i)) for i in (2, 3, 4, 5, 6, 7)]   # class sensor patch sw(patch) excluded sw(other)
        mine = before[('frozen', c)]
        assert [mine[0], mine[1], mine[2], mine[3], mine[4], mine[5]] == doc, (c, mine, doc)
BEFORE_CHECK = 'reproduces the seven rows of the addendum 3 table (class, sensor, patch, switches (patch), excluded, switches (other)) exactly'

# ---------------------------------------------------------------- AFTER: the graded binaries
AFTER_TOML = {'chair': 'chair_concept/etc/countable.toml', 'table': 'table_concept/etc/config.toml',
              'refrigerator': 'refrigerator_concept/etc/countable.toml', 'hood': 'hood_concept/etc/countable.toml',
              'nightstand': 'nightstand_concept/etc/config.toml', 'bed': 'bed_concept/etc/config.toml', 'toilet': 'toilet_concept/etc/config.toml'}
AFTER_BUILD = {c: ('table_concept' if c == 'table' else 'countable_concept') for c in CLASSES}
def charged_after(a, c):
    if a.get('unread'): return False
    if a['kind'] in ('config', 'hidden'): return a['file'] == AFTER_TOML[c]
    f = a['file']
    return bool(f) and f in SCOPE[AFTER_BUILD[c]]
MERGE = {}   # dropped after id -> kept after id (rule i)
for k in range(2533, 2541): MERGE['A%04d' % k] = None      # filled below with the per-class config row of the same key
MERGE['A2463'] = 'A2579'
by_key_cfg = collections.defaultdict(list)
for a in after:
    if a['kind'] in ('config', 'hidden') and a['key'].startswith('Tracker.') and a['key'].split('.')[1] in ('LinkPiMove', 'LinkPiGone', 'LinkLambdaNew', 'GhostEvidenceGain') and a['reader'] == 'countable_config.cpp':
        by_key_cfg[a['key']].append(a)
SITE_KEY = {'A2533': 'Tracker.LinkPiMove', 'A2534': 'Tracker.LinkPiGone', 'A2535': 'Tracker.LinkLambdaNew', 'A2536': 'Tracker.GhostEvidenceGain',
            'A2537': 'Tracker.LinkPiMove', 'A2538': 'Tracker.LinkPiGone', 'A2539': 'Tracker.LinkLambdaNew', 'A2540': 'Tracker.GhostEvidenceGain'}
byid = {a['aid']: a for a in after}
for k, v in SITE_KEY.items():
    MERGE[k] = ','.join(x['aid'] for x in by_key_cfg[v] if not x.get('unread'))
for k in MERGE: assert k in byid, k
# labels of the after rows; rules 9-13 are applied to the one first-pass row they change
def alabel(a):
    if a['aid'] in P2: return cat(P2[a['aid']][0])
    assert a['mid'] or a.get('unread'), ('unlabelled after row', a['aid'], a['key'])
    if a['mid'] == 'R057': return 'excluded'
    return cat(a['frozen'])
live = [a for a in after if a['aid'] not in MERGE and not a.get('unread')]
for a in live: a['lab'] = alabel(a)
AFTER_TABLE = {c: count([a['lab'] for a in live if charged_after(a, c)]) for c in CLASSES}

# ---------------------------------------------------------------- bookkeeping per class
baseid = {r['id']: r for r in base}
rec = {}
for c in CLASSES:
    for v in ('frozen', 'r057'):
        bp = {r['id'] for r in base if charged_before(r, c) and is_patch(blabel(r, v))}
        ball = {r['id'] for r in base if charged_before(r, c)}
        own_rows = [a for a in live if charged_after(a, c) and a['mid'] and not a['inherited']]
        retained = {a['mid'] for a in own_rows}
        any_after = {a['mid'] for a in after if a['mid'] and not a.get('unread') and a['aid'] not in MERGE}
        removed_ids = sorted(i for i in bp if i not in retained)
        gone = [i for i in removed_ids if i not in any_after]          # no after row anywhere: removed / changed / moved
        outscope = [i for i in removed_ids if i in any_after]          # still in the tree, outside this class's new binary
        scope_in = [a for a in own_rows if a['mid'] not in ball and is_patch(a['lab'])]
        relabel = [a for a in own_rows if a['mid'] in ball and is_patch(blabel(baseid[a['mid']], v)) != is_patch(a['lab'])]
        relabel_net = sum(1 for a in relabel if is_patch(a['lab'])) - sum(1 for a in relabel if not is_patch(a['lab']))
        inh = [a for a in live if charged_after(a, c) and a['inherited']]
        inh_p = [a for a in inh if is_patch(a['lab'])]
        added = [a for a in live if charged_after(a, c) and a['aid'] in P2 and is_patch(a['lab'])]
        dup = len(own_rows) - len({a['mid'] for a in own_rows})
        bpat = len(bp)
        apat = sum(1 for a in live if charged_after(a, c) and is_patch(a['lab']))
        recon = bpat - len(removed_ids) + len(scope_in) + len(inh_p) + len(added) + relabel_net
        rec[(v, c)] = dict(before=bpat, removed=len(removed_ids), gone=len(gone), outscope=len(outscope), scope_in=len(scope_in), inherited_all=len(inh),
                           inherited_patch=len(inh_p), added=len(added), relabel=relabel_net, after=apat, recon=recon, dup=dup,
                           removed_ids=removed_ids, gone_ids=gone, out_ids=outscope, added_rows=added, scope_in_rows=scope_in)

# ---------------------------------------------------------------- the 57 pass-2 patch rows
P2_PATCH = [a for a in after if a['aid'] in P2 and cat(P2[a['aid']][0]) in ('patch',) and not a.get('unread')]
assert len(P2_PATCH) == 57, len(P2_PATCH)
def part1_map(a):
    k, f, w = a['key'], a.get('file') or '', a['where']
    if k.startswith('Model.AI2CommonMode'):
        return ('G1 (shared error per view)', 'Disposition: "moved", patch to sensor? on the persistent path; pass 2 labels it patch (a floor, rule 2)', 'AI2CommonMode*')
    if k in ('literal', 'ghost_evidence_gain') and ('countable_config.cpp:430' in w or 'countable_config.h:457' in w): return ('K2b (same constant as Tracker.GhostEvidenceGain; merged into the per-class rows, not counted)', 'not a registered row', '')
    if k == 'Tracker.GhostEvidenceGain': return ('K2b ("Other new or newly counted patch rows": the legacy chair\'s Existence.EvidenceGain ported)', 'not a registered row', '')
    if 'kFrontClamp' in k: return ('baseline row B1134', 'G1 "kFrontClamp ... kept" (Disposition)', '')
    if 'front_acc re-baseline' in k: return ('baseline row B1135 (second use of the same clamp)', 'G1 "kFrontClamp ... kept" (Disposition)', '')
    if k == 'kObliquityYawGain': return ('baseline row B1142 (flag-off path only since T5)', 'G4 "kObliquityYawGain ... kept (flag-off path only)"', '')
    if k == 'obliquity cos clamp': return ('baseline row B1143', 'G4 (its cos clamp), kept', '')
    if k in ('birth top quantiles', 'birth z band', 'birth anisotropy test', 'birth size floor') or 'countable_fitter.cpp:1181' in w:
        return ('baseline row (birth seeds), context changed; A2550 is the second literal of the z-band', '"Not expected to move": birth seeds', '')
    if k == 'birth footprint percentiles': return ('baseline row R265, context changed', '"Not expected to move": birth seeds', '')
    if k == 'wall fit sigma': return ('baseline row R263, context changed', 'unmapped', '')
    if k == 'support select below_m': return ('baseline row B1148, context changed', 'unmapped', '')
    if k == 'wall segment min length': return ('baseline row (legacy %s belief), context changed' % ('fridge' if 'refrigerator' in w else 'hood'), 'unmapped', '')
    if 'countable_belief.cpp:416' in w: return ('baseline row R260 "wall segment min length", evidence line moved', 'unmapped', '')
    if 'countable_class_evidence.h:133' in w: return ('baseline row R264 "robust z-extent quantiles", evidence line changed', 'unmapped', '')
    if 'beam_channel.h:83' in w: return ('"Rows added by part 1": beam gap-ray count, moved from countable_belief.cpp:1013 (baseline "wall-hit count margin" R261)', 'unmapped (not a registered row)', '')
    if 'persistent_engine.h:116' in w: return ('"Rows added by part 1": engine volatility memory 0.995, a second copy of kLambda', '"Not expected to move": process noise', '')
    if 'persistent_engine.h:286' in w: return ('"Rows added by part 1": wall-identity gate (F1 round 6), the unflagged threshold', 'unmapped (G2 territory: wall latent, not a registered row)', '')
    if 'persistent_engine.h:1514' in w: return ('"Rows added by part 1": Sigma cap at the class prior (closed-list V5)', 'unmapped (not a registered row)', '')
    if 'standpoint_oracle.h:219' in w: return ('"Rows added by part 1": room-mask guard (K3)', 'G6 "fails by the letter"', '')
    if 'residual_risk.h:126' in w or 'residual_risk.h:114' in w or 'standpoint_oracle.h:14' in w: return ('"Rows added by part 1": residual_risk row (K3)', 'G6 "fails by the letter"', '')
    if 'countable_belief.cpp:829' in w: return ('"Rows added by part 1": birth seed-height quantile 0.95 (F1 round 6)', '"Not expected to move": birth seeds', '')
    if 'countable_birth_seed.h:44' in w: return ('file newly in scope (countable_birth_seed.h, "own:" list of After part 1)', 'unmapped (birth seeds, not a registered row)', '')
    if 'specificworker.h:250' in w: return ('"Rows added by part 1": ghost ExistenceBelief L_max copy (K2b)', 'unmapped (not a registered row)', '')
    if k.startswith('F1r9 cap switch'): return ('"Rows added by part 1": flagged threshold, gn_tol_sigma (F1 round 9), merged with A2463 tol_sigma', 'unmapped (not a registered row)', '')
    if 'chair_belief.cpp:257' in w or 'affordance_protocol.h:134' in w or 'colour_terms.h:19' in w:
        return ('baseline row (referee: "still present, not moved"); not a new constant', 'unmapped (not part 1)', '')
    return ('unmapped', 'unmapped', '')
groups = collections.OrderedDict()
for a in P2_PATCH:
    p, gm, grp = part1_map(a)
    groups.setdefault(grp or a['aid'], []).append((a, p, gm))
MERGE_NOTE_PATCH = [a for a in after if a['aid'] in MERGE and a['aid'] in P2 and cat(P2[a['aid']][0]) == 'patch']

# ---------------------------------------------------------------- render
def row6(v): return ' | '.join(str(x) for x in v)
def tot(v): return sum(v)
def ptot(v): return v[2] + v[3]
sha = hashlib.sha256(open(AREF, 'rb').read()).hexdigest()
o = []
w = o.append
w('## Addendum 4: totals before and after, with the second referee pass (2026-11-27)\n')
w('- Second pass labels: `registry/after_referee_labels_2026-11-27.tsv` in the AuRo paper folder (read-only), sha256 `%s`. 286 rows (the "context changed", "moved or removed?", new-constant and moved rows of `2026-11-27-after-referee-packet.md`), rules 9-13 in its header. Counts per label: 57 patch, 32 class, 4 sensor, 36 switch (other), 157 excluded.' % sha)
w('- After tree: `cf65a471` (`5c0029ab522691fb4747d2e7215493162f13f0b0`). Row set, ids and matches: `2026-11-27-after-count-map.tsv` (generated by `2026-11-27-after-count-map.py`). Totals: `2026-11-27-addendum4-totals.py`, which runs the map script and reads the frozen labels; no label is set here. Every label is the baseline label, referee pass 1 (`referee_labels_2026-11-27.tsv`) or referee pass 2.')
w('- The code the graded batch ran is `17f13198` (docs-only `91598c88` on top), a descendant of `cf65a471`. The difference is ten code files (`persistent_engine.h`, `wall_standoff.h`, `countable_{belief,birth_footprint,birth_seed,config,fitter}.*`, `specificworker.cpp`, tests; `git diff --stat cf65a471 17f13198`). The count is on `cf65a471`, as fixed by the after-count; rows added or changed in those ten files are not in it.')
w('')
w('### Which binary each graded class runs')
w('')
w('- Graded batch request (`docs/superpowers/results/2026-11-26-part1-grading.md:220-222`, §6): `COUNTABLE_AGENTS="refrigerator_concept hood_concept chair_concept" LEGACY_AGENTS="table_concept"`, and "`concept_agents.sh resolve` = 6 countable (fridge/hood/chair from <cls>/countable/) + 1 legacy table"; `resolve.txt` of every run = 6 countable + legacy table (`2026-11-26-part1-grading.md:101`).')
w('- Launcher at the batch code (`git show 17f13198:room_concept/estimation_scripts/scripts/concept_agents.sh`, line 54): a class runs `<cls>/countable/<cls>` with `<cls>/etc/countable.toml` only if `COUNTABLE_AGENTS` names it and `LEGACY_AGENTS` does not; lines 42-43: the table never does (the countable table is cut from part 1).')
w('- `chair_concept/countable/chair_concept` is a symlink to `../../countable_concept/bin/countable_concept` (`git cat-file -p cf65a471:chair_concept/countable/chair_concept`); `chair_concept/etc/countable.toml` exists at `cf65a471`.')
w('- So the graded chair runs **countable_concept** (not the legacy chair binary). The legacy chair rows (`chair_concept/etc/config.toml`, `chair_concept/src/*`, e.g. A2584 `chair_belief.cpp:257`) are charged to the chair BEFORE only. They are out of its new binary\'s scope, not removed from the tree.')
w('')
w('### Counting rules (auro session; the owner may overrule)')
w('')
w('1. One constant listed at several sites (toml key, loader default literal, struct default) counts once, as in the baseline\'s rule 4. Merges made (the dropped row is not counted; the kept row is):')
for k in sorted(MERGE):
    kept = MERGE[k]
    if k == 'A2463': w('   - A2463 (`persistent_engine.h:98`, `tol_sigma`, pass 2: excluded) into A2579 (`countable_belief.h:332`, `gn_tol_sigma`, pass 2: patch): one parameter, counted once, as patch.')
    else:
        a = byid[k]; w('   - %s (%s, `%s`, pass 2: %s) into the per-class config rows %s (key `%s`, one per class that runs the countable binary).' % (k, a['where'].split(' [')[0], a['key'] if a['key'] != 'literal' else SITE_KEY[k], P2[k][0], kept, SITE_KEY[k]))
w('   - Candidates NOT merged (not in the auro list; the referee labels them separately): A2550 (`countable_fitter.cpp:1181`, "second literal of one gate" with A2288, both patch), and the per-class copies of each toml key (every class has its own config file, so each class counts its own row).')
w('2. Rules 9-13 apply to both counts. The only first-pass row they change is R057 (`kEqualVolumeDilation`, sqrt(2) of a uniform, a mathematical identity, rule 9): it becomes excluded. Two baseline versions are shown: addendum 3 as frozen, and addendum 3 with R057 excluded. R057 is charged to the table only; in the after count it is excluded.')
w('3. A row is charged to a class only if it is in the build of the binary that class runs in the graded batch. AFTER: table = the legacy `table_concept` build (addendum 2 list plus the after-count deltas); chair, refrigerator, hood = the `countable_concept` build plus the class\'s own `etc/countable.toml`; nightstand, bed, toilet = `countable_concept` plus their own `etc/config.toml`. BEFORE = the addendum 2 build rule, the binaries as at baseline (the charging code of the addendum 3 script, applied to every row; %s). The 210 rows of `table_concept/etc/countable.toml` are present but unread (excluded under rule 0) and charged to no class.' % BEFORE_CHECK)
w('')
w('### Totals per class')
w('')
w('Columns as in addendum 3. "patch total" = patch + switches (patch).\n')
w('| class | version | class | sensor | patch | switches (patch) | excluded | switches (other) | patch total | total rows |')
w('|---|---|---|---|---|---|---|---|---|---|')
for c in CLASSES:
    for v, nm in (('frozen', 'before, addendum 3 as frozen'), ('r057', 'before, R057 excluded'), ('after', 'after')):
        vv = before[(v, c)] if v != 'after' else AFTER_TABLE[c]
        w('| %s | %s | %s | %d | %d |' % (c, nm, row6(vv), ptot(vv), tot(vv)))
w('')
w('### Movement of the patch total, per class')
w('')
w('"removed" = baseline patch rows (in the version shown) that no longer count for the class: no after row of that class (the row is gone, changed or its context changed: "gone"), or its file is outside the class\'s new binary ("out of scope"). "scope in" = baseline patch rows that now count for the class and did not before (own match through the class\'s new build). "inherited" = rows of the class\'s own `countable.toml` that took their label through the reader from a nightstand/bed/toilet baseline row (patch ones; all in brackets). "added" = patch rows from referee pass 2 charged to the class. "relabelled" = retained rows whose label changes patch/not patch (R057).\n')
w('| class | version | before | removed (gone + out of scope) | scope in | inherited (all) | added | relabelled | before - removed + scope in + inherited + added + relabelled | after | gap |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for c in CLASSES:
    for v in ('frozen', 'r057'):
        r = rec[(v, c)]
        w('| %s | %s | %d | -%d (%d + %d) | +%d | +%d (%d) | +%d | %+d | %d | %d | %+d |' % (c, 'frozen' if v == 'frozen' else 'R057 excluded', r['before'], r['removed'], r['gone'], r['outscope'], r['scope_in'].__len__() if isinstance(r['scope_in'], list) else r['scope_in'], r['inherited_patch'], r['inherited_all'], r['added'], r['relabel'], r['recon'], r['after'], r['after'] - r['recon']))
w('')
gaps = [(v, c, rec[(v, c)]['after'] - rec[(v, c)]['recon']) for c in CLASSES for v in ('frozen', 'r057') if rec[(v, c)]['after'] != rec[(v, c)]['recon']]
if gaps:
    w('Reconciliation gaps (not forced): ' + '; '.join('%s %s %+d' % (c, v, gp) for v, c, gp in gaps) + '.')
else:
    w('Reconciliation: before - removed + scope in + inherited + added + relabelled equals the after patch total in every class and both versions (gap 0).')
w('')
w('Duplicate baseline matches inside one class (two after rows of one class with the same baseline id): ' + ', '.join('%s %s' % (c, rec[('frozen', c)]['dup']) for c in CLASSES) + '.')
w('')
w('### Removed baseline patch rows, by class (frozen version)')
w('')
for c in CLASSES:
    r = rec[('frozen', c)]
    ids = r['removed_ids']
    if not ids: continue
    def short(i):
        b = baseid[i]; return '%s %s' % (i, b['name'])
    gone = ', '.join(short(i) for i in r['gone_ids']) or '-'
    out = ', '.join(short(i) for i in r['out_ids'][:400])
    w('- **%s**: %d gone: %s. Out of scope (%d), listed by id: %s.' % (c, len(r['gone_ids']), gone, len(r['out_ids']), ', '.join(r['out_ids']) if len(r['out_ids']) <= 400 else ', '.join(r['out_ids'][:400]) + ' ...'))
w('')
w('### The 57 patch rows of referee pass 2')
w('')
w('Each row is mapped to the part-1 item that introduced it ("Rows added by part 1", the Disposition table of "After part 1") and to the registered prediction (spec "Registered prediction for test (c)", G1-G6). The mapping is from the text of those two places; "unmapped" where neither names the row. Per-class copies of one key are one line. Charged to = classes whose graded binary contains the row.\n')
w('| ids | key | where (cf65a471) | charged to | part-1 item | registered prediction |')
w('|---|---|---|---|---|---|')
for grp, items in groups.items():
    a0, p, gm = items[0]
    ids = ', '.join(x[0]['aid'] for x in items)
    chg = sorted({c for x in items for c in CLASSES if charged_after(x[0], c)}) or ['(none: out of every graded binary)']
    w('| %s | %s | %s | %s | %s | %s |' % (ids, a0['key'].replace('|', '/'), a0['where'].split(' [')[0].replace('|', '/'), ', '.join(chg), p.replace('|', '/'), gm.replace('|', '/')))
w('')
w('Totals: %d rows in %d lines. Rows labelled patch in pass 2 and then dropped by a merge (rule 1): %s.' % (len(P2_PATCH), len(groups), ', '.join(a['aid'] for a in MERGE_NOTE_PATCH)))
w('')
w('### Notes')
w('')
w('- The chair is the largest mover: it leaves its own legacy rows and takes the countable engine, the countable.toml rows and the common/ files the countable build pulls.')
w('- The 19 "context changed" rows and the 5 "moved or removed?" rows are baseline constants that pass 2 found unchanged in role; by the definitions above they count once as removed (no match) and once as added (pass 2 patch row). Where the baseline row and the pass-2 row are charged to the same class the two cancel.')
w('- Row counts include every row of the class\'s binary (excluded rows too), as in addendum 3, so "total rows" grows with the engine files the countable binary pulls.')
w('')
sec = '\n'.join(o) + '\n\n'
s = open(DOC, encoding='utf-8').read()
a = s.index('## Addendum 4:') if '## Addendum 4:' in s else None
b = s.index('## After part 1')
s = (s[:a] if a is not None else s[:b]) + sec + '---\n\n' + s[b:]
open(DOC, 'w', encoding='utf-8').write(s)
print('written; gaps:', gaps)
for c in CLASSES:
    for v in ('frozen', 'r057'):
        r = rec[(v, c)]; print(c, v, {k: r[k] for k in ('before', 'removed', 'gone', 'outscope', 'scope_in', 'inherited_patch', 'inherited_all', 'added', 'relabel', 'after', 'recon', 'dup')} if False else (r['before'], r['removed'], len(r['scope_in_rows']), r['inherited_patch'], r['inherited_all'], r['added'], r['relabel'], r['after'], r['recon']))
    print(c, 'after row', AFTER_TABLE[c])
