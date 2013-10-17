#!/usr/bin/python

import numpy as np
import os
from yt.frontends.stream.api import load_uniform_grid


def _valid_gridfile(path, fname):
    if not os.path.isfile(os.path.join(path, fname)):
        raise OSError("Mocassin3D \"%s\" not found in %s" % (fname, path))
    return os.path.join(path, fname)


def _parse_grid0(path):
    with open(_valid_gridfile(path, 'grid0.out'), 'r') as fd:
        fd.readline()
        data = fd.readline().split()
        ddims = map(np.int64, data[:3])
        bbox = np.zeros((3,2))
        for jn, nn in enumerate(ddims):
            ax = np.zeros(nn)
            for ic in range(nn):
                ax[ic] = float(fd.readline().strip())
            bbox[jn, :] = [ax.min(), ax.max()]

    return ddims, bbox


def _parse_grid1(path, domain_dimensions):
    temp_e, dens_e, dens_h = np.loadtxt(_valid_gridfile(path, 'grid1.out'),
                                        unpack=True)
    for array in (temp_e, dens_e, dens_h):
        array.shape = tuple(domain_dimensions)
    return dict(Density=dens_h + 1e-50,
                Electron_Temperature=temp_e,
                Electron_Density=dens_e)


def _parse_plotout(path, domain_dims):
    data = {}
    lines = np.loadtxt(_valid_gridfile(path, 'plot.out'), dtype=np.float64)
    for lno in range(1, lines.shape[1]):
        data['Emission_Line_%i' % lno] = lines[:, lno].reshape(domain_dims)
    return data


def load_mocassin(path):
    if not os.path.isdir(path):
        raise OSError("%s is not a valid directory" % path)

    domain_dims, domain_edges = _parse_grid0(path)
    data = _parse_grid1(path, domain_dims)
    data.update(_parse_plotout(path, domain_dims))
    return load_uniform_grid(data, np.array(domain_dims), 1, bbox=domain_edges)


if __name__ == "__main__":
    pf = load_mocassin('/tmp/mokasyn')
    from yt.mods import ProjectionPlot
    c = pf.domain_center
    c[2] = 0.0
    for field in pf.h.field_list:
        slc = ProjectionPlot(pf, 'z', field, center=c)
        slc.save()
