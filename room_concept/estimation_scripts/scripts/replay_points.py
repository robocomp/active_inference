#!/usr/bin/env python3
"""Read the camera points and the staged LiDAR beam out of a countable replay dump (countable_concept/src/countable_replay.h).

    replay_points.py DUMP.bin --cycles C0 C1 [--out points.npz]

Prints, for the update events whose cycle is in [C0, C1], the camera point cloud (room frame) and the beam endpoints with
their labels m, and saves them (npz: cam = N x 3 float32, beam = M x 3 float64, m = M, cyc = cycle per row). Diagnosis tool
(P1 toilet bias): the format is named records — u32 event type, u64 length, payload = u32 field count, then per field u16 name
length, name, u64 value length, value; vectors are u64 count + elements; Eigen matrices are raw column-major. Little-endian.
"""
import struct, sys
import numpy as np

UPDATE_EV = 4


def record(buf, off, end):
    """{name: (offset, length)} of one named record in buf[off:end]."""
    (n,) = struct.unpack_from('<I', buf, off); off += 4
    out = {}
    for _ in range(n):
        (nl,) = struct.unpack_from('<H', buf, off); off += 2
        name = bytes(buf[off:off + nl]).decode(); off += nl
        (vl,) = struct.unpack_from('<Q', buf, off); off += 8
        out[name] = (off, vl); off += vl
    return out


def vec(buf, off, dtype, width):
    (cnt,) = struct.unpack_from('<Q', buf, off)
    return np.frombuffer(buf, dtype=dtype, count=cnt * width, offset=off + 8).reshape(cnt, width)


def main(argv):
    path = argv[0]; c0, c1 = int(argv[argv.index('--cycles') + 1]), int(argv[argv.index('--cycles') + 2])
    out = argv[argv.index('--out') + 1] if '--out' in argv else None
    buf = memoryview(open(path, 'rb').read())
    assert bytes(buf[:8]) == b'RCRPLAY\0', 'not a replay dump'
    off = 12
    cams, beams, ms, cyc = [], [], [], []
    while off + 12 <= len(buf):
        et, ln = struct.unpack_from('<IQ', buf, off); off += 12
        if et == UPDATE_EV:
            top = record(buf, off, off + ln)
            (cycle,) = struct.unpack_from('<q', buf, top['cycle'][0])
            if c0 <= cycle <= c1:
                fr = record(buf, top['frame'][0], top['frame'][0] + top['frame'][1])
                p = vec(buf, fr['points'][0], np.float32, 3)
                bm = record(buf, fr['beam'][0], fr['beam'][0] + fr['beam'][1])
                e = vec(buf, bm['endpoints'][0], np.float64, 3); m = vec(buf, bm['m'][0], np.float64, 1)[:, 0]
                cams.append(p.copy()); beams.append(e.copy()); ms.append(m.copy()); cyc.append(np.full(len(e), cycle))
        off += ln
    cam = np.concatenate(cams) if cams else np.zeros((0, 3)); beam = np.concatenate(beams) if beams else np.zeros((0, 3))
    m = np.concatenate(ms) if ms else np.zeros(0)
    print(f'{len(cams)} updates in cycles [{c0}, {c1}]: {len(cam)} camera points, {len(beam)} beam endpoints')
    if out: np.savez(out, cam=cam, beam=beam, m=m, cyc=np.concatenate(cyc) if cyc else np.zeros(0))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
