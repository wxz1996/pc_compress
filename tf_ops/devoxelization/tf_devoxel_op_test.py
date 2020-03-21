import tensorflow as tf
import numpy as np

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'voxelization'))
from tf_devox import trilinear_devoxelize

# tf.enable_eager_execution()

class GroupTriTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with self.test_session():
        feature = np.random.random((8, 3, 512)).astype(np.float32)
        pts = np.random.random((8, 3, 2048)).astype(np.float32)
        pts = tf.constant(pts)
        features = tf.constant(feature)
        res = 8
        out, inds, wgts = trilinear_devoxelize(pts, features, res)
        # print(out)
        # print(inds)
        # print(wgts)
        err = tf.test.compute_gradient_error(features, (8, 3, 512), out, (8, 3, 2048))
        print(err)



if __name__=='__main__':
  tf.test.main()
