"""Physical checks for Wannier polarization, images and cell deformation."""
import unittest

import numpy as np

from wannier_response import CENTERS, compatible_bases, matched_centers, polarization


class WannierResponseTests(unittest.TestCase):
    def setUp(self):
        self.r = 5.175*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        self.ions = np.array([[0., 0., 0.], [.25, .25, .25]])
        self.branch = -np.ones(3)

    def test_neutral_translation(self):
        reference = polarization(CENTERS, self.ions, self.r, self.branch)[2]
        shifted = polarization(CENTERS+.173, self.ions+.173, self.r, self.branch)[2]
        np.testing.assert_allclose(reference, 0., atol=1e-16)
        np.testing.assert_allclose(shifted, reference, atol=1e-16)

    def test_images_permutation_and_quantum(self):
        centers = CENTERS[[2, 0, 3, 1]].copy()
        centers += [[2, -1, 0], [-1, 2, 1], [1, 0, -2], [0, -1, 1]]
        aligned, images, order = matched_centers(centers, CENTERS, self.r)
        np.testing.assert_allclose(aligned, CENTERS)
        np.testing.assert_array_equal(order, [1, 3, 0, 2])
        np.testing.assert_array_equal(np.sum(images, axis=0), -np.array([2, 0, 0]))
        base = polarization(CENTERS, self.ions, self.r, self.branch)[2]
        translated = CENTERS.copy()
        translated[0, 0] += 1
        new = polarization(translated, self.ions, self.r, self.branch)[2]
        np.testing.assert_allclose(new-base, -2*self.r[:, 0]/np.linalg.det(self.r))

    def test_finite_difference_with_strained_cells(self):
        field = 1e-4
        chi = 11/(4*np.pi)
        pair = []
        for sign in (-1, 1):
            r = np.diag([1+sign*2e-5, 1-sign*1e-5, 1+sign*3e-5]) @ self.r
            # Four doubly occupied centers moving against the field.
            dr = np.array([0., 0., -sign*field*chi*np.linalg.det(r)/8])
            centers = CENTERS + np.linalg.solve(r, dr)
            pair.append(polarization(centers, self.ions, r, self.branch)[2])
        epsilon = np.array([0., 0., 1.])+4*np.pi*(pair[1]-pair[0])/(2*field)
        np.testing.assert_allclose(epsilon, [0., 0., 12.], atol=1e-12)

    def test_reject_cutoff_basis_change(self):
        size = compatible_bases(self.r, self.r, [3, 3, 3], 2)
        self.assertGreater(size, 0)
        with self.assertRaisesRegex(ValueError, 'G lists'):
            compatible_bases(self.r, self.r*1.1, [3, 3, 3], 2)


if __name__ == '__main__':
    unittest.main()
