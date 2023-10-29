import tensorflow as tf

#Taken from tensorflow's 2.14.0 implementation, was not available in our version of tensorflow
class RSquared(tf.keras.metrics.Metric):
  def __init__(self, name = 'r2', **kwargs):
      super(RSquared, self).__init__(**kwargs)
      self.class_aggregation = 'uniform_average'
      self.num_regressors = 0
      self.num_samples = self.add_weight(name="num_samples", dtype="int32")
      self.built = False
      self.r_squared = self.add_weight('r_squared', initializer = 'zeros')        

  def build(self, y_true_shape, y_pred_shape):
      if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
          raise ValueError(
              "R2Score expects 2D inputs with shape "
              "(batch_size, output_dim). Received input "
              f"shapes: y_pred.shape={y_pred_shape} and "
              f"y_true.shape={y_true_shape}."
          )
      if y_pred_shape[-1] is None or y_true_shape[-1] is None:
          raise ValueError(
              "R2Score expects 2D inputs with shape "
              "(batch_size, output_dim), with output_dim fully "
              "defined (not None). Received input "
              f"shapes: y_pred.shape={y_pred_shape} and "
              f"y_true.shape={y_true_shape}."
          )
      num_classes = y_pred_shape[-1]
      self.squared_sum = self.add_weight(
          name="squared_sum",
          shape=[num_classes],
          initializer="zeros",
      )
      self.sum = self.add_weight(
          name="sum",
          shape=[num_classes],
          initializer="zeros",
      )
      self.total_mse = self.add_weight(
          name="residual",
          shape=[num_classes],
          initializer="zeros",
      )
      self.count = self.add_weight(
          name="count",
          shape=[num_classes],
          initializer="zeros",
      )
      self.built = True

  def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
      y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
      if not self.built:
          self.build(y_true.shape, y_pred.shape)

      if sample_weight is None:
          sample_weight = 1

      sample_weight = tf.convert_to_tensor(sample_weight, dtype=self.dtype)
      if sample_weight.shape.rank == 1:
          # Make sure there's a features dimension
          sample_weight = tf.expand_dims(sample_weight, axis=1)
      sample_weight = tf.__internal__.ops.broadcast_weights(
          weights=sample_weight, values=y_true
      )

      weighted_y_true = y_true * sample_weight
      self.sum.assign_add(tf.reduce_sum(weighted_y_true, axis=0))
      self.squared_sum.assign_add(
          tf.reduce_sum(y_true * weighted_y_true, axis=0)
      )
      self.total_mse.assign_add(
          tf.reduce_sum((y_true - y_pred) ** 2 * sample_weight, axis=0)
      )
      self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))
      self.num_samples.assign_add(tf.size(y_true))

  def result(self):
      mean = self.sum / self.count
      total = self.squared_sum - self.sum * mean
      raw_scores = 1 - (self.total_mse / total)
      raw_scores = tf.where(tf.math.is_inf(raw_scores), 0.0, raw_scores)

      if self.class_aggregation == "uniform_average":
          r2_score = tf.reduce_mean(raw_scores)
      elif self.class_aggregation == "variance_weighted_average":
          weighted_sum = tf.reduce_sum(total * raw_scores)
          sum_of_weights = tf.reduce_sum(total)
          r2_score = weighted_sum / sum_of_weights
      else:
          r2_score = raw_scores

      if self.num_regressors != 0:
          if self.num_regressors > self.num_samples - 1:
              warnings.warn(
                  "More independent predictors than datapoints "
                  "in adjusted R2 score. Falling back to standard R2 score.",
                  stacklevel=2,
              )
          elif self.num_regressors == self.num_samples - 1:
              warnings.warn(
                  "Division by zero in Adjusted R2 score. "
                  "Falling back to standard R2 score.",
                  stacklevel=2,
              )
          else:
              n = tf.cast(self.num_samples, dtype=tf.float32)
              p = tf.cast(self.num_regressors, dtype=tf.float32)
              num = tf.multiply(
                  tf.subtract(1.0, r2_score), tf.subtract(n, 1.0)
              )
              den = tf.subtract(tf.subtract(n, p), 1.0)
              r2_score = tf.subtract(1.0, tf.divide(num, den))
      return r2_score

  def reset_state(self):
      for v in self.variables:
          v.assign(tf.zeros(v.shape, dtype=v.dtype))