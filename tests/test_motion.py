import unittest
from pathlib import Path
import numpy as np
from motion import derivative, Recording


class MeasurementsTest(unittest.TestCase):
    def test_rolling_window_bounds_and_statistics(self):
        r = Recording(next(Path(__file__).resolve().parents[1].glob('*.c3d')))
        for seconds in [3,5,15]:
            start,end,summary = r.rolling_summary(2400,seconds)
            self.assertEqual((start,end),(20-seconds,20))
            expected = r.summary(20-seconds,20)
            self.assertEqual(summary,expected)
        start,end,_ = r.rolling_summary(120,15)
        self.assertEqual((start,end),(0,1))
        start,end,summary = r.rolling_summary(0,3)
        self.assertEqual((start,end),(0,0))
        self.assertTrue(np.isnan(summary['Walking speed (m/s)']))
        for seconds in [2,16,float('nan')]:
            with self.assertRaises(ValueError):
                r.rolling_summary(120,seconds)

    def test_constant_velocity(self):
        t = np.arange(100)/100
        points = np.column_stack((2*t, -3*t, t*0))
        actual = derivative(points,100,5)
        np.testing.assert_allclose(actual[5:-5],np.tile([2,-3,0],(90,1)),atol=1e-12)
        self.assertTrue(np.isnan(actual[:5]).all())
        self.assertTrue(np.isnan(actual[-5:]).all())

    def test_missing_data_invalidates_whole_derivative_window(self):
        points = np.column_stack((np.arange(100)/100,np.zeros(100),np.zeros(100)))
        points[50] = np.nan
        actual = derivative(points,100,5)
        self.assertTrue(np.isnan(actual[45:56]).all())
        self.assertAlmostEqual(actual[44,0],1)
        self.assertAlmostEqual(actual[56,0],1)

    def test_relative_translation_removal(self):
        t = np.arange(100)/100
        pelvis = np.column_stack((2*t,t,t*0))
        wrist = pelvis + [0,0,1]
        np.testing.assert_allclose(derivative(wrist-pelvis,100,5)[5:-5],0)

    def test_supplied_recording(self):
        path = next(Path(__file__).resolve().parents[1].glob('*.c3d'))
        r = Recording(path)
        self.assertEqual(r.xyz.shape,(8126,50,3))
        self.assertEqual(r.rate,120)
        self.assertLess(r.coverage[r.lookup['LAnkleOut']],.02)
        self.assertTrue(np.isnan(r.xyz[~r.valid]).all())
        summary = r.summary(5,15)
        self.assertTrue(0 <= summary['Walking speed (m/s)'] < 5)
        self.assertTrue(0 <= summary['Left arm range (deg)'] <= 180)
        self.assertTrue(np.isfinite(r.speed).mean() > .99)
        with self.assertRaises(ValueError):
            r.summary(10,5)


if __name__ == '__main__':
    unittest.main()
