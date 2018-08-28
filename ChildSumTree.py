import tensorflow as tf

class ChildSumTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
  """LSTM with two state inputs.

  This is the model described in section 3.2 of 'Improved Semantic
  Representations From Tree-Structured Long Short-Term Memory
  Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
  dropout as described in 'Recurrent Dropout without Memory Loss'
  <http://arxiv.org/pdf/1603.05118.pdf>.
  """

  def __init__(self, num_units, keep_prob=1.0):
    """Initialize the cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      keep_prob: Keep probability for recurrent dropout.
    """
    super(BinaryTreeLSTMCell, self).__init__(num_units)
    self._keep_prob = keep_prob

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      c_list = []
      h_list = []
      h_sum = None 
      for child in state: 
        c, h = child
        c_list.append(c)
        h_list.append(h)
        if h_sum is None:
          h_sum = h
        else:
          h_sum = tf.add(h_sum, h)

      kernel_i, kernel_f, kernel_j, kernel_o = tf.split(value=self._kernel, num_or_size_splits=4, axis=1)
      bias_i, bias_f, bias_j, bias_o = tf.split(value=self._bias, num_or_size_splits=4, axis=0)
      
      one = constant_op.constant(1, dtype=dtypes.int32)

      input_h_sum_concat = array_ops.concat([inputs, h_sum], 1)
      
      gate_inputs_i = math_ops.matmul(input_h_sum_concat, kernel_i)
      gate_inputs_i = nn_ops.bias_add(gate_inputs_i, bias_i)

      gate_inputs_j = math_ops.matmul(input_h_sum_concat, kernel_j)
      gate_inputs_j = nn_ops.bias_add(gate_inputs_j, bias_j)

      gate_inputs_o = math_ops.matmul(input_h_sum_concat, kernel_o)
      gate_inputs_o = nn_ops.bias_add(gate_inputs_o, bias_o)

      add = math_ops.add
      multiply = math_ops.multiply
      sigmoid = math_ops.sigmoid
      forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
      
      c_f_sum = None
      for k in range(len(c_list)):
        gate_inputs_f_k = math_ops.matmul(array_ops.concat([inputs, h_list[k]], 1), kernel_f)
        gate_inputs_f_k = nn_ops.bias_add(gate_inputs_f_k, bias_f)
        if c_f_sum is None: 
          c_f_sum = multiply(sigmoid(add(gate_inputs_f_k, forget_bias_tensor)), c_list[k])
        else:
          f_k_sig = multiply(sigmoid(add(gate_inputs_f_k, forget_bias_tensor)), c_list[k]) 
          c_f_sum = add(c_f_sum, f_k_sig)

      
      new_c = add(c_f_sum, multiply(sigmoid(gate_inputs_i), self._activation(gate_inputs_j)))
      new_h = multiply(self._activation(new_c), sigmoid(gate_inputs_o))

      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

      return new_h, new_state


