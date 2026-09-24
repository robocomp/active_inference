#!/usr/bin/env python3
"""
migrate_agent.py <agent> [--write] — move one agent's config reads onto rc::cfg::Reader.

WHAT IT REWRITES. Three of the fleet's read dialects, in place, keeping every call site's shape so
the diff is one added argument per line rather than a rewrite:

  A  out.field = getf("Key", def)                 ->  getf("Key", def, "<description>")
     (the four local lambdas become forwarders onto the registry; 11 agents share them byte-identically)
  B  rc::ConfigLoaderUtils::load_optional<T,L>(cfg, "Key", tgt)  ->  cfgr.opt<T,L>("Key", tgt, "...")
  C  load_optional("Key", tgt) / load_optional_cast<T>("Key", tgt)  ->  cfgr.opt<...>("Key", tgt, "...")

WHERE THE DESCRIPTIONS COME FROM, in priority order:

  1. THE AGENT'S OWN HEADER. `out.state_eps = getf("TableConcept.StateEps", 0.04f)` is matched to
     `float state_eps = 0.04f;  // <doc>` in *_config.h / *_params.h / specificworker.h. This is the
     author's own sentence, written beside the field, and it is the best source in the tree: for
     table_concept it covered 118 of 130 keys.
  2. THE TOML COMMENT, filtered. Config prose is often RATIONALE rather than definition ("was 0.03",
     "⚠ NOT MEASURED ON SHADOW"), and it can be edited to match a paste and go on reading plausibly —
     door_concept's config carries four chair-shaped Tracker.BirthSeat* keys whose comment was
     reworded to sound door-ish and which door_concept reads nowhere. So a comment is REFUSED when it
     opens with ⚠/★/was/TODO/NOTE/FIXME, or runs past 110 characters.
  3. NOTHING. An empty description is legal and COUNTED — the startup table prints "N undocumented",
     which is the backlog made visible rather than a gap nobody can see.

This is a migration aid, not an oracle: every description it lifts is still the author's text about
that field, but the MATCH is mechanical. Read the diff.
"""
from __future__ import annotations
import argparse, glob, os, re, sys

REFUSE = re.compile(r'^\s*(?:⚠|★|was\b|TODO|NOTE|FIXME|#|-)', re.I)

FIELD_DECL = re.compile(
    r'^\s*(?:static\s+)?(?:constexpr\s+)?'
    r'(?:float|double|int|bool|std::string|std::size_t|size_t|std::uint\w+|uint\w+|unsigned)\s+'
    r'(\w+)\s*(?:=|;|\{)')


def first_sentence(text: str) -> str:
    """The definition, not the whole essay: cut at the first sentence end or at 150 characters."""
    t = re.sub(r'\s+', ' ', text).strip()
    m = re.search(r'(?<=[.;])\s+(?=[A-Z\u2605\u26a0])', t)
    if m and m.start() >= 24:
        t = t[:m.start()]
    if len(t) > 150:
        # Cut at a WORD boundary. "...don't integ" is not a shorter description, it is a broken one.
        cut = t[:150].rfind(' ')
        t = t[:cut if cut > 60 else 150].rstrip() + '…'
    return t.rstrip(' .,;')


def header_docs(agent_dir: str) -> dict[str, str]:
    """field name -> the author's own one-liner, from the headers beside the code."""
    docs: dict[str, str] = {}
    for h in sorted(glob.glob(os.path.join(agent_dir, 'src', '*.h'))):
        block: list[str] = []
        for line in open(h, errors='ignore'):
            st = line.strip()
            if st.startswith('//'):
                t = st.lstrip('/ ').strip()
                # Drop SECTION BANNERS ("── Top/leg SDF split band ─────"): they title a group of
                # fields, so attaching one to the first field of the group mislabels it.
                if '───' in t or t.count('─') > 3:
                    continue
                if t and not set(t) <= set('─-=★⚠ '):
                    block.append(t)
                continue
            m = FIELD_DECL.match(line)
            if m:
                inline = line.split('//', 1)[1].strip() if '//' in line else ''
                # ★The FIRST sentence of the block, not the last LINE of it. A comment block states
                # what the field is and then qualifies it, so its tail is a fragment: cabinet's
                # sigma_obs came out as "face is attributed to the slab (candidate) vs a leg" and
                # masks_stall_timeout_ms as "...never trips it. 0 = disable the gate" - both true,
                # both unreadable on their own. Rejoin the wrapped lines, then cut at the sentence.
                doc = inline or first_sentence(' '.join(block))
                if doc and m.group(1) not in docs:
                    docs[m.group(1)] = doc
                block = []
            elif st:
                block = []
    return docs


def toml_docs(agent_dir: str) -> dict[str, str]:
    """dotted key -> its config comment, only where the comment is a DEFINITION."""
    out: dict[str, str] = {}
    path = os.path.join(agent_dir, 'etc', 'config.toml')
    if not os.path.exists(path):
        return out
    section, block = '', []
    for raw in open(path, errors='ignore'):
        line = raw.rstrip('\n')
        st = line.strip()
        if not st:
            block = []
            continue
        if st.startswith('#'):
            block.append(st.lstrip('# ').strip())
            continue
        m = re.match(r'^\s*\[([^\]]+)\]', line)
        if m:
            section, block = m.group(1), []
            continue
        m = re.match(r'^\s*([A-Za-z_]\w*)\s*=\s*(.*)$', line)
        if m:
            inline = m.group(2).split('#', 1)[1].strip() if '#' in m.group(2) else ''
            cand = inline or (block[-1] if block else '')
            key = f'{section}.{m.group(1)}' if section else m.group(1)
            if cand and not REFUSE.match(cand) and len(cand) <= 110:
                out[key] = cand
            block = []
    return out


def clean(text: str) -> str:
    """One line, no quotes that would break the literal, no trailing clutter."""
    t = re.sub(r'\s+', ' ', text).strip().rstrip('.,;')
    t = t.replace('\\', '/').replace('"', "'")
    return t[:150]


def migrate(agent: str, root: str, write: bool) -> int:
    d = os.path.join(root, agent)
    hdocs, tdocs = header_docs(d), toml_docs(d)
    total = lifted_h = lifted_t = blank = 0
    touched: list[str] = []

    for src in sorted(glob.glob(os.path.join(d, 'src', '*.cpp'))):
        text = original = open(src, errors='ignore').read()

        # ── the four lambdas become forwarders ───────────────────────────────────────────────────
        # ★Replace the lambdas ONE AT A TIME, not as a fixed quartet. viewer3d and three others
        # declare them in a different order (gets last, not getb), so a pattern anchored on "getf
        # first, getb last" silently matched nothing there - leaving old lambdas beside new call
        # signatures, which is four compile errors and, worse, four agents that looked migrated.
        LAMBDA = re.compile(r'[ \t]*auto get([fisb]) = \[&\]\([^)]*\)[^{]*\{.*?\n[ \t]*\};\n', re.S)
        FWD = {
            'f': '    const auto getf = [&](std::string_view k, float def, std::string_view what,\n'
                 '                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };\n',
            'i': '    const auto geti = [&](std::string_view k, int def, std::string_view what,\n'
                 '                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };\n',
            's': '    const auto gets = [&](std::string_view k, std::string def, std::string_view what,\n'
                 '                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };\n',
            'b': '    const auto getb = [&](std::string_view k, bool def, std::string_view what,\n'
                 '                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };\n',
        }
        first = LAMBDA.search(text)
        if first:
            text = LAMBDA.sub(lambda m: FWD[m.group(1)], text)
            text = text[:first.start()] + (
                '    // Every read below registers its key, its CODE DEFAULT and a one-line description,\n'
                '    // so the startup table can say where each value came from - not just what it is.\n'
                '    // (The four local lambdas now forward to the shared registry; the call sites are\n'
                '    // unchanged except for that description.)\n'
                '    rc::cfg::Reader reader(cfg, "' + agent + '");\n'
            ) + text[first.start():]

        # ── dialect A call sites ─────────────────────────────────────────────────────────────────
        # ★The assignment prefix is CAPTURED, never recovered by searching the match for "get".
        # Seven field names in this fleet contain that substring - ai2_view_budget, masks_target_frame
        # and residual's four grid_forget_* ablation flags - so `m.group(0).index('get')` split
        # "ai2_view_budget = getf(" inside the WORD and produced `ai2_view_budgetf(`. Two agents failed
        # to compile and the rest were only luckier, not safer.
        CALL = re.compile(r'((?:\w+)\.(\w+)\s*=\s*)?\bget([fisb])\(\s*"([^"]+)"\s*,\s*([^();]*?)\s*\)')

        def repl(m):
            nonlocal total, lifted_h, lifted_t, blank
            head, field, fn, key, default = m.groups()
            head = head or ''
            total += 1
            doc = ''
            if field and field in hdocs:
                doc = clean(hdocs[field]); lifted_h += 1
            elif key in tdocs:
                doc = clean(tdocs[key]); lifted_t += 1
            else:
                blank += 1
            return f'{head}get{fn}("{key}", {default},\n@@"{doc}")'

        text = CALL.sub(repl, text)

        # resolve continuation indents
        if '@@' in text:
            fixed, indent = [], 0
            for line in text.split('\n'):
                if line.startswith('@@'):
                    fixed.append(' ' * (indent + 8) + line[2:])
                else:
                    indent = len(line) - len(line.lstrip())
                    fixed.append(line)
            text = '\n'.join(fixed)

        if text != original:
            touched.append(os.path.relpath(src, root))
            if write:
                if '#include "../../common/config_report/config_read.h"' not in text:
                    text = re.sub(r'(#include "[^"]*config\.h"\n)',
                                  r'\1\n#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)\n',
                                  text, count=1)
                open(src, 'w').write(text)

    cov = 100 * (lifted_h + lifted_t) // max(total, 1)
    print(f'{agent:22} {total:4} sites  ·  {lifted_h:4} from header  {lifted_t:3} from toml  '
          f'{blank:3} blank  ·  {cov:3}% documented'
          + ('' if write else '   [dry run]'))
    for t in touched:
        print(f'                       {t}')
    return total


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('agents', nargs='+')
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    for ag in a.agents:
        migrate(ag, a.root, a.write)
