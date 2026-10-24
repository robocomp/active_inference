#!/usr/bin/env python3
"""Canned run folders for recognition_report.py: a passing set (exit 0) and one with a duplicate room node (exit 1).
   python3 -I test_recognition_report.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'recognition_report.py')

def memory(names, extra_passages=True):
    sym, nid = {}, 100
    for nm in names:
        nid += 1
        sym[str(nid)] = dict(id=str(nid), name=nm, type='room', attribute={}, links=[])
    rooms = {n['name']: k for k, n in sym.items()}
    def face(name, room):
        nonlocal nid
        nid += 1
        sym[str(nid)] = dict(id=str(nid), name=name, type='metaconcept', attribute={}, links=[])
        sym[rooms[room]]['links'].append(dict(label='RT', src=rooms[room], dst=str(nid),
                                               linkAttribute={'rt_translation': {'type': 3, 'value': [1.0, 2.0, 0.0]}}))
        return str(nid)
    if extra_passages:
        for pn, ra, rb in (('passage_0', 'r1_room_1', 'r2_room_2'), ('passage_1', 'r1_room_1', 'r3_room_3'),
                           ('passage_2', 'r2_room_2', 'r3_room_3')):
            fa, fb = face(pn + '_a', ra), face(pn + '_b', rb)
            nid += 1
            sym[str(nid)] = dict(id=str(nid), name=pn, type='metaconcept', links=[
                dict(label='match', src=str(nid), dst=fa), dict(label='match', src=str(nid), dst=fb)],
                attribute={'object_subtype': {'type': 0, 'value': 'passage'}})
    return {'DSRModel': {'symbols': sym}}

LTSM = """[recog] episode begins: provisional room_2 behind room_1
[recog] recognition node posted: 1 candidate room(s)
[recog] episode begins: provisional room_3 behind room_1
[recog] episode begins: provisional room_4 behind room_3
[recog] SWAP requested: room_id 2 (posterior 0.973)
[swap] STAGED: room_4 is r2_room_2 (case A); late recall under the crossed door
[swap] COMMITTED: left r2_room_2; nothing staged in memory
[recall] recall_room_3 posted under door_5: behind it lies remembered r3_room_3 (passage face a)
[recall] recall_room_1 posted under door_2: behind it lies remembered r1_room_1 (passage face a)
"""
# line counts per leg: ltsm 0-3 leg1, 3-3 leg2 ... made by the lists below
def legs_tsv(ltsm_cuts):
    rows = ['leg\tstatus\tt0_ms\tt1_ms\tltsm0\tltsm1\tdoor0\tdoor1\troom0\troom1']
    t = 1000
    for i in range(6):
        rows.append(f'{i+1}\tOK\t{t}\t{t+1000}\t{ltsm_cuts[i]}\t{ltsm_cuts[i+1]}\t0\t100\t0\t100'); t += 1000
    return '\n'.join(rows) + '\n'

DOOR = """door_concept: [restore] provisional 'room_4' dropped — ltsm recognised the room as 'room_2' (late recall); its identity rows are deleted
door_concept: [restore] 'room_2' recalled (late) — its remembered doors are restored once its outline loads
"""
def gt_csv(ok=True):
    rows = ['ts_ms,gt_x,gt_y,est_x,est_y,cov_xx']
    for i in range(0, 6000, 100):
        t = 1000 + i
        e = 0.0 if ok else 0.5
        rows.append(f'{t},1.0,1.0,{1.0 + (0.01 if ok else e)},1.0,0.0004')
    return '\n'.join(rows) + '\n'

def make_run(d, names, ok=True):
    os.makedirs(os.path.join(d, 'logs'))
    # legs: 1 -> lines 0-2, 2 -> 2-2, 3 -> 2-3, 4 -> 3-7, 5 -> 7-8, 6 -> 8-9
    open(os.path.join(d, 'legs.tsv'), 'w').write(legs_tsv([0, 2, 2, 3, 7, 8, 9]))
    open(os.path.join(d, 'logs', 'ltsm_agent.log'), 'w').write(LTSM)
    open(os.path.join(d, 'logs', 'door_concept.log'), 'w').write(DOOR)
    open(os.path.join(d, 'logs', 'room_concept.log'), 'w').write('[room][late-recall] room_2 adopted from memory\n')
    json.dump(memory(names), open(os.path.join(d, 'ltsm_memory.json'), 'w'))
    open(os.path.join(d, 'gt_error.csv'), 'w').write(gt_csv(ok))

def run(d):
    p = subprocess.run([sys.executable, '-I', REPORT, d], capture_output=True, text=True)
    return p.returncode, p.stdout

def main():
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, 'good'); make_run(good, ['r1_room_1', 'r2_room_2', 'r3_room_3'])
        rc, out = run(good)
        assert rc == 0, (rc, out)
        assert 'PASS C3' in out and 'FAIL' not in out, out
        dup = os.path.join(t, 'dup'); make_run(dup, ['r1_room_1', 'r2_room_2', 'r3_room_3', 'r9_room_2'])
        rc, out = run(dup)
        assert rc == 1, (rc, out)
        assert 'FAIL C3' in out and 'room_2' in out, out
        bad = os.path.join(t, 'badpose'); make_run(bad, ['r1_room_1', 'r2_room_2', 'r3_room_3'], ok=False)
        rc, out = run(bad)
        assert rc == 1 and 'FAIL C4' in out, (rc, out)
    print('test_recognition_report: OK')

if __name__ == '__main__': main()
