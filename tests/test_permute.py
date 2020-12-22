#!/usr/bin/env python

import numpy as np
from improc3d.permute import permute3d


def test_permute():
    image = np.random.rand(90, 80, 100)
    perm, inv_x, inv_y, inv_z = permute3d(image, x=2, y=0, z=1)
    assert perm.shape == (100, 90, 80)
    assert np.array_equal(np.transpose(image, [2, 0, 1]), perm)
    image2 = permute3d(perm, inv_x, inv_y, inv_z, return_inv_axes=False)
    assert np.array_equal(image, image2)
    print('successful.')


if __name__ == '__main__':
    test_permute()
