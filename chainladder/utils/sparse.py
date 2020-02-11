# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import sparse
from chainladder import ARRAY_BACKEND
from sparse import COO as sp


sp.isnan = np.isnan
sp.newaxis = np.newaxis
sp.array = np.array
sp.where = np.where
sp.nan = np.nan
sp.testing = np.testing
sp.nansum = sparse.nansum
sp.concatenate = sparse.concatenate
sp.diagonal = sparse.diagonal
sp.zeros = sparse.zeros
sp.ones = np.ones
sp.testing.assert_array_equal = np.testing.assert_equal

def nan_to_num(a):
    a.fill_value = 0
    return sparse.COO(a)
sp.nan_to_num = nan_to_num


def expand_dims(a, axis):
    a = a.copy()
    shape = [slice(None, None)] * a.ndim
    if axis == -1:
        shape.append(None)
    elif axis < -1:
        axis = axis + 1
        shape.insert(axis, None)
    else:
        shape.insert(axis, None)
    return a.__getitem__(tuple(shape))
sp.expand_dims = expand_dims

def arange(*args, **kwargs):
    return sparse.COO.from_numpy(np.arange(*args, **kwargs))
sp.arange = arange

def cumsum(a, axis=None, dtype=None, out=None):
    return sparse.COO.from_numpy(np.cumsum(a.todense(), axis=axis, dtype=dtype, out=out))
sp.cumsum = cumsum

def swapaxes(a, axis1, axis2):
    l = []
    for item in range(a.ndim):
        if item == axis1:
            l.append(axis2)
        elif item == axis2:
            l.append(axis1)
        else:
            l.append(item)
    print(l)
    a.coords = a.coords[l,:]
    return a

sp.swapaxes = swapaxes
