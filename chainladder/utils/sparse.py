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
sp.nansum = sparse.nansum
sp.concatenate = sparse.concatenate
sp.diagonal = sparse.diagonal
sp.zeros = sparse.zeros

def nan_to_num(a):
    return a
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
