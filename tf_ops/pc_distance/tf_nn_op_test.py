import tensorflow as tf
import numpy as np
from tf_nn_distance import nn_distance

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with self.test_session():
        pts_1 = np.random.random((4, 128, 3)).astype(np.float32)
        pts_2 = np.random.random((4, 128, 3)).astype(np.float32)
        pts_1 = tf.constant(pts_1)
        pts_2 = tf.constant(pts_2)
        reta, retb, retc, retd = nn_distance(pts_1, pts_2)
        err = tf.test.compute_gradient_error(pts_1, (4, 128, 3), reta, (4, 128))
        print(err)

if __name__=='__main__':
  tf.test.main()
