#!/usr/bin/env python3
"""Tests for mesh_extents.py.  Run: python3 tools/gt_mesh/test_mesh_extents.py   (needs the Webots asset cache, see mesh_extents.py)

Every expected number comes from somewhere other than the code under test:
  * bed mattress 1.571 +- 0.001 m: measured 11-22 from Bed.proto's mattress.obj (the brief of Task 9); the length 1.971 and the
    mattress box centre are from the same measurement. The tolerance is the brief's.
  * toilet 0.738 x 0.589, z 0.002 .. 1.013, centre (-0.066, -0.009): Task 1 (docs/superpowers/results/2026-11-24-step0.md), a
    3-decimal table, so the tolerance is half of the last digit, 0.0005 m.
  * hood canopy 0.5 x 1.296 m, z 0 .. 0.3: the eight points written in the .wbt itself (x +-0.25, y +-0.648, z 0 / 0.3), exact.
  * table top 0.8 x 1.8 m, z 0.6014 .. 0.7298 and the chair seat: the table1.dae vertices and the Chair.proto SEAT vertices; the chair is
    cross-checked against the proto's own collision seat Box (0.5 x 0.5 at z 0.4 .. 0.5), which the proto writes to the centimetre, so
    the visual seat must agree with it to within 0.01 m.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mask_error'))
import mesh_extents as M

WBT = os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt')


def near(a, b, tol, what):
    assert abs(a - b) <= tol, f'{what}: {a:.5f} vs {b:.5f} (tol {tol})'


def test_largest_z_gap_splits_a_slab_from_its_legs():
    v = np.array([[0, 0, 0.0], [1, 0, 0.1], [0, 1, 0.2], [0, 0, 1.0], [1, 1, 1.05]])
    assert len(M.split_at_largest_z_gap(v, 'low')[0]) == 3 and len(M.split_at_largest_z_gap(v, 'high')[0]) == 2


def test_split_refuses_a_mesh_without_a_layer_gap():
    v = np.array([[0, 0, z] for z in (0.0, 0.1, 0.2, 0.35, 0.5)])      # gaps 0.1, 0.1, 0.15, 0.15: ratio 1
    try: M.split_at_largest_z_gap(v, 'low')
    except ValueError as e: assert 'largest' in str(e) and 'second' in str(e)
    else: raise AssertionError('no ValueError')
    v2 = np.array([[0, 0, z] for z in (0.0, 0.1, 0.2, 0.39)])          # 0.19 vs 0.1: ratio 1.9 < 2
    try: M.split_at_largest_z_gap(v2, 'low')
    except ValueError: pass
    else: raise AssertionError('ratio 1.9 accepted')


def test_the_real_splits_are_layer_gaps():
    n = M.graded(WBT, 'Table1')[6]
    for cls, node in (('chair', None), ('table', n)):
        gaps = M.layer(cls, node, WBT)[1]
        assert gaps[0][2] >= M.GAP_RATIO * gaps[1][2], (cls, gaps)


def test_bed_is_the_mattress():
    n, c, x, y, z, yaw, node = M.graded(WBT, 'BED_1')
    lx, ly, z0, z1, ox, oy = M.layer_box('bed', node, WBT)
    near(min(lx, ly), 1.571, 0.001, 'mattress width'); near(max(lx, ly), 1.971, 0.001, 'mattress length')
    near(ox, 0.0, 0.001, 'mattress centre x'); near(oy, 0.0, 0.001, 'mattress centre y')


def test_toilet_matches_task1():
    lx, ly, z0, z1, ox, oy = M.layer_box('toilet')
    for got, want, w in ((lx, 0.738, 'lx'), (ly, 0.589, 'ly'), (z0, 0.002, 'z0'), (z1, 1.013, 'z1'), (ox, -0.066, 'ox'), (oy, -0.009, 'oy')):
        near(got, want, 0.0005, 'toilet ' + w)


def test_hood_canopy_is_the_wbt_frustum_and_the_origin_is_its_centre():
    n, c, x, y, z, yaw, node = M.graded(WBT, 'HOOD_0')
    lx, ly, z0, z1, ox, oy = M.layer_box('hood', node, WBT)
    for got, want in ((lx, 0.5), (ly, 1.296), (z0, 0.0), (z1, 0.3), (ox, 0.0), (oy, 0.0)): near(got, want, 1e-9, 'hood')


def test_table_top_and_chair_seat():
    n, c, x, y, z, yaw, node = M.graded(WBT, 'Table1')
    lx, ly, z0, z1, ox, oy = M.layer_box('table', node, WBT)
    near(lx, 0.8, 0.001, 'table top x'); near(ly, 1.8, 0.001, 'table top y'); near(z0, 0.6014, 0.0001, 'table top z0')
    lx, ly, z0, z1, ox, oy = M.layer_box('chair')
    near(lx, 0.5, 0.01, 'seat x'); near(ly, 0.5, 0.01, 'seat y'); near(ox, 0.0, 0.01, 'seat cx'); near(oy, 0.0, 0.01, 'seat cy')
    assert 0.38 < z0 < z1 < 0.52, (z0, z1)


def test_def_and_name_lookups_agree_and_world_centre_is_rotated():
    a = M.graded(WBT, 'TOILET_1'); b = M.graded(WBT, 'Toilet')
    assert a[2:6] == b[2:6]
    lay = M.world_layers(WBT)['Toilet']
    # toilet yaw -pi/2: the object-frame centre (-0.066, -0.009) lands at (x + oy', y - ox') ... recomputed independently
    c, s = math.cos(a[5]), math.sin(a[5]); ox, oy = lay['box'][4], lay['box'][5]
    near(lay['cx'], a[2] + c * ox - s * oy, 1e-12, 'cx'); near(lay['cy'], a[3] + s * ox + c * oy, 1e-12, 'cy')


def test_one_source_of_known_extents():
    import measure_mask_error as MM
    assert MM.known_extents is M.known_extents and MM.obj_extents is M.obj_extents
    assert M.known_extents() is M.known_extents() and set(M.known_extents()) == {'toilet', 'bed'}
    here = os.path.dirname(M.__file__)
    for f in ('../mask_error/measure_mask_error.py', '../step0/nees_persistent.py'):
        src = open(os.path.join(here, f)).read()
        assert 'KNOWN_EXTENTS' not in src and 'from mesh_extents import' in src, f


if __name__ == '__main__':
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            try: f(); print('ok  ', k)
            except Exception as e: fails += 1; print('FAIL', k, '--', repr(e))
    print(f'{fails} failed'); sys.exit(1 if fails else 0)
