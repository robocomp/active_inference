import sys, re, collections
DOC, CNT, REF = sys.argv[1:4]
L = open(DOC, encoding='utf-8').read().split('\n')
# stop at "After part 1"
end = next(i for i,l in enumerate(L) if l.startswith('## After part 1'))
L = L[:end]
# per-target file lists
targets = {}; cur=None; d=None; inblk=False
for i,l in enumerate(L):
    m = re.match(r'<details><summary>(\w+): \d+ files</summary>', l)
    if m: cur=m.group(1); targets[cur]=set(); continue
    if l.startswith('### Newly counted files'): cur=None
    if cur is None: continue
    m = re.match(r'\*\*(.+?/)\*\* \(\d+\)', l)
    if m: d=m.group(1); continue
    if l.startswith('```'): inblk = not inblk; continue
    if inblk and l.strip(): targets[cur].add(d + l.strip())
# rows with '?' labels, with section path, in document order
h2=h3=h4=None; rows=[]
for i,l in enumerate(L):
    if l.startswith('## '): h2=l[3:].strip(); h3=h4=None; continue
    if l.startswith('### '): h3=l[4:].strip(); h4=None; continue
    if l.startswith('#### '): h4=l[5:].strip(); continue
    if not l.startswith('|') or l.startswith('|---'): continue
    c=[x.strip() for x in l.strip().strip('|').split('|')]
    if not c[-1].endswith('?') or c[-1].startswith('ambig'): continue
    sec=' / '.join(x for x in (h2,h3,h4) if x)
    where = next((x for x in c if re.search(r'[\w/]+\.(h|cpp|toml)(:\d+)?', x)), '')
    rows.append(dict(sec=sec, name=c[0] if 'Own-directory' not in sec else c[1], cls=c[0], where=where, best=c[-1], line=i+1))
cnt=[l.rstrip('\n').split('\t') for l in open(CNT, encoding='utf-8')]
ref={}
for l in open(REF, encoding='utf-8'):
    if l.startswith('#') or l.startswith('id\t'): continue
    p=l.rstrip('\n').split('\t'); ref[p[0]]=p[1]
print('doc ? rows', len(rows), 'counter rows', len(cnt), 'ref', len(ref))
# align in order
bad=0
for r,c in zip(rows,cnt):
    if not (c[1]==r['sec'] and c[3]==r['best']): bad+=1; print('MISMATCH', r['line'], r['sec'], r['name'], '|', c)
    r['id']=c[0]; r['ref']=ref[c[0]]
print('mismatches', bad)
CLS = {'chair':'chair_concept','table':'table_concept','refrigerator':'refrigerator_concept','hood':'hood_concept',
       'nightstand':'countable_concept','bed':'countable_concept','toilet':'countable_concept'}
def owner(r):
    s=r['sec']
    for k in CLS:
        if s.startswith(k+' ') or s.startswith(k+' (') or s==k: return ('own',k)
    if 'Own-directory' in s: return ('own', r['cls'])
    if s.startswith('countable engine'): return ('engine',None)
    return ('common', None)
def files_of(r):
    return re.findall(r'(common/[\w/.\-]+\.(?:h|cpp|hpp))', r['where'])
res={}
for k,t in CLS.items():
    old=collections.Counter(); new=collections.Counter(); ids=[]
    for r in rows:
        kind,o=owner(r)
        if kind=='own': inc = (o==k)
        elif kind=='engine': inc = (t=='countable_concept')
        else:
            fs=files_of(r)
            if not fs: print('NOFILE', r['id'], r['sec'], r['where']); inc=False
            else: inc = any(f in targets[t] for f in fs)
        if inc: old[r['best'].rstrip('?')]+=1; new[r['ref']]+=1; ids.append(r['id'])
    res[k]=(old,new,ids)
    print(k, sum(old.values()), dict(old), '->', dict(new))
import json; json.dump({k:(dict(o),dict(n),i) for k,(o,n,i) in res.items()}, open(sys.argv[4],'w'))
