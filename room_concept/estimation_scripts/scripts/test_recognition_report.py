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

SWAPL = "[recog] SWAP requested: room_id 2 (posterior 0.973; metric log_inside 6.66, appearance loglr 1.20, new: metric log_new -97.56)"
LTSM = """[recog] episode begins: provisional room_2 behind room_1
[recog] recognition node posted: 1 candidate room(s)
[recog] episode begins: provisional room_3 behind room_1
[recog] episode begins: provisional room_4 behind room_3
[recog] SWAP requested: room_id 2 (posterior 0.973; metric log_inside 6.66, appearance loglr 1.20, new: metric log_new -97.56)
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
REAL_HDR = "ts_ms,gt_x,gt_y,gt_theta,est_x,est_y,est_theta,gt_theta_raw,sdf_mse,iters,cov_tt,imu_dtheta,wheel_dtheta,wheel_shadow_dtheta,imu_segs,wheel_segs,pred_x,pred_y,pred_theta,dx_local,dy_local,calib_k_v,calib_k_w,calib_yaw,calib_eps,calib_carried,calib_dropped,calib_sig_kv,calib_sig_kw,calib_sig_yaw,calib_pos_var,calib_b_omega,calib_informed,calib_cond,imu_dvx,imu_dvy,wheel_dvx,wheel_dvy,imu_dpx,imu_dpy,imu_lin_segs,fb_ts,fb_cal_x,fb_cal_y,fb_cal_th,fb_nom_x,fb_nom_y,fb_nom_th,fb_corr_pitch,fb_corr_height,fb_corr_yaw,gt_x_world,gt_y_world"
def gt_csv(ok=True, cov_xx=None):
    """the REAL gt_error.csv header (copied from crossing/vlm_live1): cov_tt only; cov_xx is added only when asked"""
    cols = REAL_HDR.split(',') + (['cov_xx'] if cov_xx else [])
    rows = [','.join(cols)]
    for i in range(0, 6000, 100):
        v = {c: '0' for c in cols}
        v.update(ts_ms=str(1000 + i), gt_x='1.0', gt_y='1.0', est_x=str(1.0 + (0.01 if ok else 0.5)), est_y='1.0', cov_tt='0.0001')
        if cov_xx: v['cov_xx'] = str(cov_xx)
        rows.append(','.join(v[c] for c in cols))
    return '\n'.join(rows) + '\n'

def trace_csv(first_ok=True):
    h = 'stamp_ms,provisional_id,act,room_id,candidates'
    rows = [h]
    for t in (1100, 1500, 3100):                      # legs 1 and 3 steps: new wins
        pn = 0.9 if first_ok else 0.2
        rows.append(f'{t},9,None,0,2:{1 - pn:.2f}:-5:0;new:{pn}::-1')
    rows.append('4500,9,Swap,2,2:0.97:6.6:1.2;new:0.03::-97')
    return '\n'.join(rows) + '\n'

def make_run(d, names, ok=True, ltsm=LTSM, door=DOOR, cov_xx=None, cuts=(0, 2, 2, 3, 7, 8, 9), trace=True, first_ok=True):
    os.makedirs(os.path.join(d, 'logs'))
    open(os.path.join(d, 'legs.tsv'), 'w').write(legs_tsv(list(cuts)))
    open(os.path.join(d, 'logs', 'ltsm_agent.log'), 'w').write(ltsm)
    open(os.path.join(d, 'logs', 'door_concept.log'), 'w').write(door)
    open(os.path.join(d, 'logs', 'room_concept.log'), 'w').write('[room][late-recall] room_2 adopted from memory\n')
    json.dump(memory(names), open(os.path.join(d, 'ltsm_memory.json'), 'w'))
    open(os.path.join(d, 'gt_error.csv'), 'w').write(gt_csv(ok, cov_xx))
    if trace: open(os.path.join(d, 'recognition_trace.csv'), 'w').write(trace_csv(first_ok))

def run(d):
    p = subprocess.run([sys.executable, '-I', REPORT, d], capture_output=True, text=True)
    return p.returncode, p.stdout

def case(t, name, expect_rc, must, **kw):
    d = os.path.join(t, name); make_run(d, kw.pop('names', ['r1_room_1', 'r2_room_2', 'r3_room_3']), **kw)
    rc, out = run(d)
    assert rc == expect_rc, (name, rc, out)
    for m in must: assert m in out, (name, m, out)
    return out

def main():
    GOOD = ['r1_room_1', 'r2_room_2', 'r3_room_3']
    with tempfile.TemporaryDirectory() as t:
        out = case(t, 'good', 0, ['PASS C1', 'PASS C2', 'PASS C3', 'NOT MEASURED C4', 'PASS C5a', 'NOT MEASURED C5b', 'PASS C6', 'PASS C7', 'PASS C8',
                                  '7 PASS, 0 FAIL, 2 NOT MEASURED'])
        assert 'FAIL' not in out.replace('0 FAIL', ''), out
        assert 'NOT MEASURED C4' in out and 'PASS C4' not in out, out        # the real header has no xy covariance
        case(t, 'notrace', 0, ['NOT MEASURED C1'], trace=False)
        case(t, 'dup', 1, ['FAIL C3', 'duplicate name(s) room_2'], names=GOOD + ['r9_room_2'])
        case(t, 'nees_ok', 0, ['PASS C4'], cov_xx=0.0004)
        case(t, 'nees_bad', 1, ['FAIL C4'], ok=False, cov_xx=0.0004)
        case(t, 'c1_swap_leg1', 1, ['FAIL C1'], ltsm=SWAPL + '\n' + LTSM, cuts=(0, 3, 3, 4, 8, 9, 10))
        case(t, 'c1_map_lost', 1, ['FAIL C1'], first_ok=False)
        case(t, 'c2_two_swaps', 1, ['FAIL C2', '2 Swap'], ltsm=LTSM.replace('[swap] COMMITTED', SWAPL + '\n[swap] COMMITTED'), cuts=(0, 2, 2, 3, 8, 9, 10))
        case(t, 'c2_no_contrib', 0, ['NOT MEASURED C2'], ltsm=LTSM.replace('; metric log_inside 6.66, appearance loglr 1.20, new: metric log_new -97.56', ''))
        case(t, 'c7_undo', 1, ['FAIL C7', '1 Undo'], ltsm=LTSM + '[recog] UNDO requested: room_id 2 contradicted\n', cuts=(0, 2, 2, 3, 7, 8, 10))
        case(t, 'c8_fresh', 1, ["FAIL C8", 'FRESH'], door=DOOR + "door_concept: [restore] 'room_2' entered FRESH (x) — nothing restored\n")
        case(t, 'c8_nodrop', 1, ['FAIL C8'], door=DOOR.splitlines()[1] + '\n')
        # leg-4 boundary: the Swap/STAGED lines landed AFTER the leg-4 window (cut at 3) => leg 4 has none, leg 5 has a Swap
        case(t, 'boundary_late', 1, ['FAIL C2', '0 Swap', 'FAIL C6'], cuts=(0, 2, 2, 3, 3, 8, 9))
    # leg-5 door from ltsm's swap-time association line, windowed to after the leg-4 start
    with tempfile.TemporaryDirectory() as t:
        lg = os.path.join(t, 'l.log')
        def door(text, frm=0):
            open(lg, 'w').write(text)
            return subprocess.run([sys.executable, '-I', REPORT, '--leg5-door', lg, str(frm)], capture_output=True, text=True).stdout.strip()
        old = '[swap] crossed door r1_door_1 <-> K face r2_door_1\n'
        new = '[swap] crossed door r3_door_4 <-> K face r2_door_7\n'
        assert door(old + new) == 'door_7'
        assert door(old + new, 1) == 'door_7' and door(old + new, 2) == ''      # windowed: an older line is not this swap's
        assert door('[swap] crossed door r3_door_4 <-> K face new\n') == ''      # new face => discovery fails
        assert door('nothing\n') == ''
    print('test_recognition_report: OK')

if __name__ == '__main__': main()
