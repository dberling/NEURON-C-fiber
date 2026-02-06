import numpy as np
from neuron import h
from lfpykit import CellGeometry, LineSourcePotential

def cellgeometry_from_neuron_pt3d(fiber_secs_ordered, save_geometry):
    """
    Convert a NEURON model with explicit pt3d geometry
    into an lfpykit.CellGeometry object.
    """

    xs, ys, zs, ds = [], [], [], []

    for sec in fiber_secs_ordered:
        nseg = sec.nseg
        n3d = int(h.n3d(sec=sec))

        # Extract pt3d points
        x3d = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
        y3d = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
        z3d = np.array([h.z3d(i, sec=sec) for i in range(n3d)])
        d3d = np.array([h.diam3d(i, sec=sec) for i in range(n3d)])

        # Arc length along section
        arc = np.zeros(n3d)
        arc[1:] = np.cumsum(np.sqrt(
            np.diff(x3d)**2 +
            np.diff(y3d)**2 +
            np.diff(z3d)**2
        ))
        arc /= arc[-1]

        # Segment boundaries in normalized arc length
        seg_edges = np.linspace(0, 1, nseg + 1)

        for i in range(nseg):
            s0, s1 = seg_edges[i], seg_edges[i + 1]
            sm = 0.5 * (s0 + s1)

            x0 = np.interp(s0, arc, x3d)
            y0 = np.interp(s0, arc, y3d)
            z0 = np.interp(s0, arc, z3d)

            x1 = np.interp(s1, arc, x3d)
            y1 = np.interp(s1, arc, y3d)
            z1 = np.interp(s1, arc, z3d)

            dmid = np.interp(sm, arc, d3d)

            xs.append([x0, x1])
            ys.append([y0, y1])
            zs.append([z0, z1])
            ds.append(dmid)
    if save_geometry:
        np.save('lfpy_geom_xyz.npy', np.array([xs, ys, zs]))
        np.save('lfpy_geom_d.npy', np.array(ds))

    return CellGeometry(
        x=np.asarray(xs),
        y=np.asarray(ys),
        z=np.asarray(zs),
        d=np.asarray(ds),
    )

def get_ERM(fiber_secs_ordered, electr_xyz_um, cond_SPERm, save_geometry=False):
    # port fiber geometry
    cell = cellgeometry_from_neuron_pt3d(fiber_secs_ordered, save_geometry)
    # set up electrode
    lsp = LineSourcePotential(
        cell,
        x=np.array([electr_xyz_um[0]]),
        y=np.array([electr_xyz_um[1]]),
        z=np.array([electr_xyz_um[2]]),
        sigma=cond_SPERm
    )
    ERM = lsp.get_transformation_matrix()
    return ERM