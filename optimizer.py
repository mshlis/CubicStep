import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.keras.layers import *
from tensorflow.python.keras import backend
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras import initializers
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
import functools
import six

class CubicStep(optimizer_v2.OptimizerV2):
    def __init__(self, 
                 learning_rate=1e-3,
                 warmup_steps=5,
                 **kwargs):
        super().__init__(**kwargs)
        self._lr = learning_rate
        self._lr_t = None
        self._warmup_steps = warmup_steps
        self._warmup_steps_t = None

    def _prepare(self, var_list):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._warmup_steps_t = tf.convert_to_tensor(self._warmup_steps, name="warmup_steps")

        
    def add_single_slot(self, slot_name, shape, dtype, initializer='zeros'):
        slot_dict = self._slots
        weight = slot_dict.get(slot_name, None)
        if weight is None:
            if isinstance(initializer, six.string_types) or callable(initializer):
                initializer = initializers.get(initializer)
                initial_value = functools.partial(
                        initializer, shape=shape, dtype=dtype)
            else:
                initial_value = initializer

            weight = tf_variables.Variable(
                name="%s" % (slot_name),  # pylint: disable=protected-access
                dtype=dtype,
                trainable=False,
                initial_value=initial_value)

            backend.track_variable(weight)
            slot_dict[slot_name] = weight
            self._restore_single_slot_variable(
              slot_name=slot_name,
              slot_variable=weight)
            self._weights.append(weight)
        return weight
    
    def _restore_single_slot_variable(self, slot_name, slot_variable):
        """Restore a newly created slot variable's value."""
        deferred_restorations = self._deferred_slot_restorations.get(
            slot_name, [])

        for checkpoint_position in deferred_restorations:
            checkpoint_position.restore(slot_variable)
    
    def _create_slots(self, var_list):
        self.add_single_slot('pre_loss', shape=(), dtype=tf.float32)
        for var in var_list:
            self.add_slot(var, 'pre_weight')
            self.add_slot(var, 'pre_grad')
            self.add_slot(var, 'step_count')

    def _resource_apply_dense(self, grad, var):
        # get variables
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        loss = tf.cast(self._loss, var.dtype.base_dtype)
        pre_loss = self._slots['pre_loss']
        pre_weight = self.get_slot(var, 'pre_weight')
        pre_grad = self.get_slot(var, 'pre_grad')
        step_count = self.get_slot(var, 'step_count')
        warmup_steps = tf.cast(self._warmup_steps_t, step_count.dtype.base_dtype)
        
        # tensor setup
        ph = tf.ones_like(var)
        sgd_var = var - lr_t*grad
        _pre_loss = pre_loss*ph
        _loss = loss*ph
        poly_out = tf.stack([_loss, _pre_loss, grad, pre_grad], axis=-1)

        # solving the 4 linear equations
        var_shape = var.shape #tuple(list(var.shape)[1:])
        def inv(X, w0, w1):
            _Z = tf.reshape(tf.eye(4), (1,)*(len(X.shape)-2)+(4,4))
            _Z = tf.tile(_Z, tf.concat([tf.shape(X)[:-2], tf.constant([1,1], dtype=tf.int32)],0))
            Z = tf.split(_Z, num_or_size_splits=4, axis=-2)
            w0 = w0[...,tf.newaxis, tf.newaxis]
            w1 = w1[...,tf.newaxis, tf.newaxis]

            Z[3] += -(2/(w0-w1))*(Z[0]-Z[1]) + Z[2]
            Z[3] /= (w1-w0)**2

            Z[2] += -(1/(w0-w1))*(Z[0]-Z[1]) - (2*w0**2 - w0*w1 - w1**2)*Z[3]
            Z[2] /= (w0-w1)

            Z[1] += -Z[0] - (w1**2-w0**2)*Z[2] - (w1**3-w0**3)*Z[3]
            Z[1] /= (w1-w0)

            Z[0] += -w0*Z[1] - w0**2*Z[2] - w0**3*Z[3]
            return tf.concat(Z, -2)
              
        Q = [[var**k for k in range(4)],
             [pre_weight**k for k in range(4)],
             [k*var**(max(0,k-1)) for k in range(4)],
             [k*pre_weight**(max(0,k-1)) for k in range(4)]]
        Q = tf.stack([tf.stack(_q,-1) for _q in Q], -2)
        
        """
        Q_shapelen = len(Q.shape)
        perm = list(range(Q_shapelen))
        tmp = perm[-1]
        perm[-1] = perm[-2]
        perm[-2] = tmp
        Q = tf.transpose(Q, perm=perm)
        Qbad = tf.random.normal(mean=0.0, stddev=1, shape=tf.shape(Q))
        Q = tf.where(tf.tile(tf.abs(var - pre_weight)[...,tf.newaxis, tf.newaxis], (1,)*len(var.shape)+(4,4))<=1e-3, Qbad, Q)
        Qinv = tf.linalg.inv(Q)
        """
        
        Qinv = inv(Q, var, pre_weight)
        poly_out = tf.reshape(poly_out, (-1,4))
        Qinv = tf.reshape(Qinv, (-1,4,4))
        #poly_coeffs = tf.map_fn(lambda x: tf.matmul(x[0],x[1]), (Qinv, poly_out[...,tf.newaxis]), dtype=tf.float32) 
        poly_coeffs = tf.keras.backend.batch_dot(Qinv, poly_out, axes=[-1,-1])
        poly_coeffs = tf.reshape(poly_coeffs, var_shape+(4,))

        # solving for critical points
        root_a = -poly_coeffs[...,2]
        root_b_base = (poly_coeffs[...,2]**2 - 3*poly_coeffs[...,1]*poly_coeffs[...,3])
        root_b = tf.where(root_b_base > 0, root_b_base**.5, 0)
        root_scale = 1 / (3*poly_coeffs[...,3] + 1e-8)
        roots = [root_scale*(root_a + root_b), root_scale*(root_a - root_b)]

        # min max text 
        min_v_max = [poly_coeffs[...,2] + 3*poly_coeffs[...,3]*root for root in roots]
        new_var = tf.where(min_v_max[0] > 0, roots[0], roots[1])
        
        # stability safety
        new_var = tf.maximum(tf.minimum(pre_weight, var), new_var)
        new_var = tf.minimum(tf.maximum(pre_weight, var), new_var)
        
        cond_a = tf.greater(var, pre_weight)
        cond_b = tf.less(-grad,0)
        cond_c = tf.greater(-pre_grad,0)
        cond_pos = tf.math.logical_and(tf.math.logical_and(cond_a,cond_b),cond_c)
        cond_neg = tf.math.logical_and(tf.math.logical_and(tf.math.logical_not(cond_a),
                                                           tf.math.logical_not(cond_b)),
                                       tf.math.logical_not(cond_c))
        cond = tf.math.logical_or(cond_pos, cond_neg)
        
        #new_var = .5*(var+pre_weight) ##### ERRASE #########
        
        new_var = tf.where(root_b_base >= 0, new_var, sgd_var)
        new_var = tf.where(cond, new_var, sgd_var)
        #new_var = tf.where(tf.abs(poly_coeffs[...,3]) >= 1e-3, new_var, sgd_var)
        new_var = tf.where(tf.reduce_sum(tf.cast(tf.abs(poly_coeffs[...,1:]) >= 1e4, tf.int32), -1)>0, sgd_var, new_var)
        new_var = tf.where(tf.abs(var - pre_weight)<=1e-6, sgd_var, new_var)

        #new_var = sgd_var
        
        step_bool = tf.cast(step_count < warmup_steps, tf.float32)
        new_var = sgd_var*step_bool + new_var*(1-step_bool)
    
        # grouping updates
        updates = []
        updates.append(state_ops.assign(pre_weight, var))
        updates.append(state_ops.assign(step_count, step_count+1))
        updates.append(state_ops.assign(var, new_var))
        updates.append(state_ops.assign(pre_loss, loss))
        return tf.group(*updates)
    
    def apply_gradients(self, loss, *args, **kwargs):
        self._loss = loss
        super().apply_gradients(*args, **kwargs)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("No sparse implementation")
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': backend.get_value(self._lr_t),
            'warmup_steps': backend.get_value(self._warmup_steps_t)
        })
        return config