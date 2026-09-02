#!/usr/bin/env python3
"""Lift PairObs + VertexBlock + Accum verbatim out of mount_lidar_pair.h into a standalone header,
so the test compiles with neither DSR nor Qt. Verbatim on purpose: a test against a COPY of the
algorithm tests the copy."""
import pathlib
src = pathlib.Path(__file__).resolve().parents[2] / 'src' / 'mount_lidar_pair.h'
s = src.read_text()
a = s.index('    /// One corner seen by both sensors')
b = s.index('    /// Builds the pose-free pair')
c = s.index('    /// ── THE PER-VERTEX OFFSET NUISANCE')
d = s.index('}   // namespace rc::mount')
out = pathlib.Path(__file__).with_name('vb.h')
out.write_text("#pragma once\n#include <Eigen/Dense>\n#include <map>\n#include <algorithm>\n"
               "namespace rc::mount {\n" + (s[a:b] + s[c:d]).replace('IMAGE_EDGE_NUISANCES', '5') + "\n}\n")
print(f"wrote {out}")
