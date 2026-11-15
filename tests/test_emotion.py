import unittest
from pathlib import Path
import numpy as np
from motion import Recording
from emotion_model import INFO, map_markers, prepare_window, fill_short_gaps, EmotionModel


class EmotionAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Recording(next(Path(__file__).resolve().parents[1].glob('*.c3d')))
        cls.joints = map_markers(cls.r)

    def test_mapping_and_model_shape(self):
        self.assertEqual(self.joints.shape,(8126,21,3))
        # Heel mapping remains observed even when the ankle marker is absent.
        np.testing.assert_equal(self.joints[:,3],self.r.marker('LHeel'))
        x = prepare_window(self.joints[1200:1801],120)
        self.assertEqual(x.shape,(1,48,21,3,1))
        self.assertTrue(np.isfinite(x).all())
        frame = x[0,0,:,:,0]
        for j,p in enumerate(INFO['parents'][1:],1):
            self.assertAlmostEqual(np.linalg.norm(frame[j]-frame[p]),INFO['template_bone_lengths'][j])

    def test_room_translation_rotation_invariance(self):
        source = self.joints[1200:1801]
        angle=.8
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        moved = source@rotation.T+[10,-5,2]
        np.testing.assert_allclose(prepare_window(source,120),prepare_window(moved,120),atol=1e-10)

    def test_gap_limits(self):
        points = self.joints[1200:1801].copy()
        points[100:110,2] = np.nan
        self.assertTrue(np.isfinite(fill_short_gaps(points,120)).all())
        points[100:125,2] = np.nan
        with self.assertRaises(ValueError):
            fill_short_gaps(points,120)
        points = self.joints[1200:1801].copy()
        points[0,2] = np.nan
        with self.assertRaises(ValueError):
            fill_short_gaps(points,120)

    def test_published_checkpoint_inference_and_gates(self):
        m=EmotionModel()
        incomplete=m.predict(self.r,self.joints,120,5)
        self.assertIsNone(incomplete['scores'])
        a=m.predict(self.r,self.joints,1800,5)
        b=m.predict(self.r,self.joints,1800,5)
        self.assertEqual(a['scores'],b['scores'])
        self.assertEqual(list(a['scores']),['Happy','Sad','Angry','Neutral'])
        self.assertTrue(all(0 <= v <= 1 for v in a['scores'].values()))
        broken=self.joints.copy()
        broken[1200:1801,2] = np.nan
        self.assertIsNone(m.predict(self.r,broken,1800,5)['scores'])


if __name__=='__main__':
    unittest.main()
