import unittest
from pathlib import Path
import numpy as np
from motion import Recording
from protective_model import ProtectiveModel, map_markers


def numpy_reference(model, windows):
    """Independent channels-last implementation of the published pose branch."""
    w = {k:v.numpy().astype(np.float64) for k,v in model.weights.items()}
    def softmax(x, axis=-1):
        e = np.exp(x-x.max(axis=axis,keepdims=True))
        return e/e.sum(axis=axis,keepdims=True)
    def conv(x, name):
        kernel = w[name+'/kernel:0']
        slices = np.lib.stride_tricks.sliding_window_view(x,len(kernel),axis=1)
        return np.einsum('ntck,kco->nto',slices,kernel)+w[name+'/bias:0']
    def bn(x,name):
        return (x-w[name+'/moving_mean:0'])/np.sqrt(w[name+'/moving_variance:0']+.001)*w[name+'/gamma:0']+w[name+'/beta:0']
    def dense(x,name):
        return x@w[f'{name}/{name}/kernel:0']+w[f'{name}/{name}/bias:0']
    parts = []
    for j in range(22):
        x = np.maximum(bn(conv(windows[:,:,j],'model_15/conv1d_140'),'model_15/batch_normalization_20'),0)
        x = np.tanh(bn(conv(x,'model_15/conv1d_141'),'model_15/batch_normalization_21'))
        name = f'conv1d_{146+j}'
        attention = softmax(conv(x,f'{name}/{name}'),axis=1)
        parts.append((x*attention).sum(axis=1))
    data = np.stack(parts,axis=-1)
    a = softmax(dense(np.tanh(dense(data,'dense_10')),'bodyattention'))
    return softmax(dense((data*a).reshape(len(data),-1),'suboutput1'))


class ProtectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Recording(next(Path(__file__).resolve().parents[1].glob('*.c3d')))
        cls.p = map_markers(cls.r)
        cls.m = ProtectiveModel()

    def test_numpy_reference_parity(self):
        windows = self.p[1200:1560:2][None]
        np.testing.assert_allclose(self.m.infer(windows),numpy_reference(self.m,windows),atol=2e-5)

    def test_time_sampling_and_aggregation(self):
        result = self.m.predict(self.r,self.p,1800,5)
        windows = np.stack([self.p[start:start+360:2] for start in [1200,1320,1440]])
        expected = self.m.infer(windows).mean(axis=0)
        np.testing.assert_allclose(list(result['scores'].values()),expected,atol=1e-6)
        self.assertEqual(result['subwindows'],3)
        self.assertAlmostEqual(sum(result['scores'].values()),1,places=6)
        self.assertEqual(result,self.m.predict(self.r,self.p,1800,5))
        self.assertIsNotNone(self.m.predict(self.r,self.p,2400,15)['scores'])

    def test_missing_data_and_partial_window(self):
        self.assertIsNone(self.m.predict(self.r,self.p,120,3)['scores'])
        broken = self.p.copy()
        broken[1300:1400,0] = np.nan
        self.assertIsNone(self.m.predict(self.r,broken,1800,5)['scores'])
        with self.assertRaises(ValueError):
            self.m.infer(np.full((1,180,22,3),np.nan))


if __name__ == '__main__':
    unittest.main()
