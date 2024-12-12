"""
MIT License

Copyright (c) 2024 Xiaohang Ma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import numpy as np
import tensorflow as tf
# from keras import ops
from keras.layers import Layer
# from keras import layers

custom_module = tf.load_op_library(os.path.join(os.getcwd(), 'crfasrnn_keras', 'src','cpp', 'high_dim_filter.so'))
# custom_module = tf.load_op_library(os.path.join( 'cpp', 'high_dim_filter.so'))
from tensorflow.python.framework import ops
@ops.RegisterGradient('HighDimFilter')
def _high_dim_filter_grad(op, grad):
    """ Gradients for the HighDimFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (RGB values of the image).

    Args:
    op: The `high_dim_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `high_dim_filter` op.

    Returns:
    Gradients with respect to the input of `high_dim_filter`.
    """

    rgb = op.inputs[1]
    grad_vals = custom_module.high_dim_filter(grad, rgb,
                                              bilateral=op.get_attr('bilateral'),
                                              theta_alpha=op.get_attr('theta_alpha'),
                                              theta_beta=op.get_attr('theta_beta'),
                                              theta_gamma=op.get_attr('theta_gamma'),
                                              backwards=True)

    return [grad_vals, tf.zeros_like(rgb)]

def _diagonal_initializer(shape, *ignored, **ignored_too):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape, *ignored, **ignored_too):
    return -1 * _diagonal_initializer(shape)

class HiddenCrfRnnLayer(Layer):
    def __init__(self, image_dims, num_classes, theta_alpha, theta_beta, theta_gamma,
                 num_iterations, trainable=True, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.trainable = trainable
        super(HiddenCrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.compatibility_matrix = self.add_weight(
            name='compatibility_matrix',
            shape=(self.num_classes, self.num_classes),
            initializer=_potts_model_initializer,
            trainable=self.trainable
        )
        self.local_global_potential = self.add_weight(
            name='local_global_potential',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer,
            trainable=self.trainable
        )
        self.spatial_ker_weights = self.add_weight(
            name='spatial_ker_weights',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer,
            trainable=self.trainable
        )
        self.bilateral_ker_weights = self.add_weight(
            name='bilateral_ker_weights',
            shape=(self.num_classes, self.num_classes),
            initializer=_diagonal_initializer,
            trainable=self.trainable
        )
        super(HiddenCrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unary_potentials, feature_map = inputs
        unary_potentials = tf.transpose(unary_potentials[0], perm=(2, 0, 1))

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, feature_map, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, feature_map, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)

        elbos = []

        for y in range(self.num_classes):
            q_values = unary_potentials

            for _ in range(self.num_iterations):
                softmax_q = tf.nn.softmax(q_values, axis=0)

                # Spatial filtering
                spatial_out = custom_module.high_dim_filter(softmax_q, feature_map, bilateral=False,
                                                            theta_gamma=self.theta_gamma)
                spatial_out = spatial_out / spatial_norm_vals
                spatial_out = tf.matmul(self.spatial_ker_weights, tf.reshape(spatial_out, (self.num_classes, -1)))

                # Bilateral filtering
                bilateral_out = custom_module.high_dim_filter(softmax_q, feature_map, bilateral=True,
                                                              theta_alpha=self.theta_alpha,
                                                              theta_beta=self.theta_beta)
                bilateral_out = bilateral_out / bilateral_norm_vals
                bilateral_out = tf.matmul(self.bilateral_ker_weights, tf.reshape(bilateral_out, (self.num_classes, -1)))

                # Compatibility transform
                message_passing = tf.reshape(spatial_out + bilateral_out, (c, h, w))
                pairwise = tf.matmul(self.compatibility_matrix, tf.reshape(message_passing, (c, -1)))
                pairwise = tf.reshape(pairwise, (c, h, w))

                # Adding unary and local-global potentials
                local_global_potential = tf.einsum('ij,jkl->ikl', self.local_global_potential, self.one_hot_label(y, h, w))
                q_values = unary_potentials + local_global_potential - pairwise

            # Compute ELBO for this label y
            elbo = self.compute_elbo(q_values, unary_potentials, y, h, w, softmax_q, feature_map, spatial_norm_vals, bilateral_norm_vals)
            elbos.append(elbo)

        return tf.stack(elbos)

    def one_hot_label(self, y, h, w):
        label = np.zeros((self.num_classes, h, w), dtype=np.float32)
        label[y, :, :] = 1.0
        return tf.convert_to_tensor(label)

    def compute_elbo(self, q_values, unary_potentials, y, h, w, q_softmax, feature_map, spatial_norm_vals, bilateral_norm_vals):
        # Unary term contribution
        unary_term = tf.reduce_sum(q_softmax * unary_potentials)

        # Local-global potential contribution
        local_global_potential = tf.convert_to_tensor(self.one_hot_label(y, h, w), dtype=tf.float32)
        local_global_term = tf.reduce_sum(
            tf.einsum('ij,jkl->ikl', self.local_global_potential, local_global_potential) * q_softmax
        )

        # Pairwise term contribution
        spatial_out = custom_module.high_dim_filter(q_softmax, feature_map, bilateral=False, theta_gamma=self.theta_gamma)
        spatial_out = spatial_out / spatial_norm_vals
        spatial_out = tf.matmul(self.spatial_ker_weights, tf.reshape(spatial_out, (self.num_classes, -1)))

        bilateral_out = custom_module.high_dim_filter(q_softmax, feature_map, bilateral=True, 
                                                      theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
        bilateral_out = bilateral_out / bilateral_norm_vals
        bilateral_out = tf.matmul(self.bilateral_ker_weights, tf.reshape(bilateral_out, (self.num_classes, -1)))

        pairwise_term = -tf.reduce_sum(
            (spatial_out + bilateral_out) * tf.reshape(q_softmax, (self.num_classes, -1))
        )

        # Entropy term
        entropy = -tf.reduce_sum(q_softmax * tf.math.log(q_softmax + 1e-10))

        elbo = unary_term + local_global_term + entropy + pairwise_term
        return elbo


# class HiddenCrfRnnLayer(Layer):
#     def __init__(self, image_dims, num_classes, theta_alpha, theta_beta, theta_gamma,
#                  num_iterations, trainable=True, **kwargs):
#         self.image_dims = image_dims
#         self.num_classes = num_classes
#         self.theta_alpha = theta_alpha
#         self.theta_beta = theta_beta
#         self.theta_gamma = theta_gamma
#         self.num_iterations = num_iterations
#         self.trainable = trainable
#         super(HiddenCrfRnnLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.compatibility_matrix = self.add_weight(
#             name='compatibility_matrix',
#             shape=(self.num_classes, self.num_classes),
#             initializer='zeros',
#             trainable=self.trainable
#         )
#         self.local_global_potential = self.add_weight(
#             name='local_global_potential',
#             shape=(self.num_classes, self.num_classes),
#             initializer='zeros',
#             trainable=self.trainable
#         )
#         super(HiddenCrfRnnLayer, self).build(input_shape)

#     def call(self, inputs):
#         unary_potentials, feature_map = inputs
#         unary_potentials = tf.transpose(unary_potentials[0], perm=(2, 0, 1))

#         c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
#         all_ones = np.ones((c, h, w), dtype=np.float32)

#         # Prepare filter normalization coefficients
#         spatial_norm_vals = custom_module.high_dim_filter(all_ones, feature_map, bilateral=False,
#                                                           theta_gamma=self.theta_gamma)
#         bilateral_norm_vals = custom_module.high_dim_filter(all_ones, feature_map, bilateral=True,
#                                                             theta_alpha=self.theta_alpha,
#                                                             theta_beta=self.theta_beta)

#         elbos = []

#         for y in range(self.num_classes):
#             q_values = unary_potentials

#             for _ in range(self.num_iterations):
#                 softmax_q = tf.nn.softmax(q_values, axis=0)

#                 # Spatial filtering
#                 spatial_out = custom_module.high_dim_filter(softmax_q, feature_map, bilateral=False,
#                                                             theta_gamma=self.theta_gamma)
#                 spatial_out = spatial_out / spatial_norm_vals

#                 # Bilateral filtering
#                 bilateral_out = custom_module.high_dim_filter(softmax_q, feature_map, bilateral=True,
#                                                               theta_alpha=self.theta_alpha,
#                                                               theta_beta=self.theta_beta)
#                 bilateral_out = bilateral_out / bilateral_norm_vals

#                 # Compatibility transform
#                 message_passing = tf.matmul(self.compatibility_matrix, tf.reshape(spatial_out + bilateral_out, (c, -1)))
#                 pairwise = tf.reshape(message_passing, (c, h, w))

#                 # Adding unary and local-global potentials
#                 local_global_potential = tf.einsum('ij,jkl->ikl', self.local_global_potential, self.one_hot_label(y, h, w))
#                 q_values = unary_potentials + local_global_potential - pairwise

#             # Compute ELBO for this label y
#             elbo = self.compute_elbo(q_values, unary_potentials, y, h, w, softmax_q, feature_map, spatial_norm_vals, bilateral_norm_vals)
#             elbos.append(elbo)

#         return tf.stack(elbos)

#     def one_hot_label(self, y, h, w):
#         label = np.zeros((self.num_classes, h, w), dtype=np.float32)
#         label[y, :, :] = 1.0
#         return tf.convert_to_tensor(label)

#     def compute_elbo(self, q_values, unary_potentials, y, h, w, q_softmax, feature_map, spatial_norm_vals, bilateral_norm_vals):
#         # Unary term contribution
#         unary_term = tf.reduce_sum(q_softmax * unary_potentials)

#         # Local-global potential contribution
#         local_global_potential = tf.convert_to_tensor(self.one_hot_label(y, h, w), dtype=tf.float32)
#         local_global_term = tf.reduce_sum(
#             tf.einsum('ij,jkl->ikl', self.local_global_potential, local_global_potential) * q_softmax
#         )

#         # Pairwise term contribution
#         spatial_out = custom_module.high_dim_filter(q_softmax, feature_map, bilateral=False, theta_gamma=self.theta_gamma)
#         spatial_out = spatial_out / spatial_norm_vals

#         bilateral_out = custom_module.high_dim_filter(q_softmax, feature_map, bilateral=True, 
#                                                       theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
#         bilateral_out = bilateral_out / bilateral_norm_vals

#         pairwise_term = -tf.reduce_sum((spatial_out + bilateral_out) * q_softmax)

#         # Entropy term
#         entropy = -tf.reduce_sum(q_softmax * tf.math.log(q_softmax + 1e-10))

#         elbo = unary_term + local_global_term + entropy + pairwise_term
#         return elbo

class Hcrf_VI_Posterior(Layer):
    """ Fixed point iteration seeking the optimal variational posterior distribution.
    We adapt the fast Gaussian Filter described in:
    
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1))
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1))

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape


# class ELBO(Layer):
#     """ Compute the ELBO as the approximation of log-likelihood
#     We adapt the fast Gaussian Filter described in:
    
#     Conditional Random Fields as Recurrent Neural Networks,
#     S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
#     ICCV 2015
#     """

#     def __init__(self, image_dims, num_classes,
#                  theta_alpha, theta_beta, theta_gamma,
#                  num_iterations, **kwargs):
#         self.image_dims = image_dims
#         self.num_classes = num_classes
#         self.theta_alpha = theta_alpha
#         self.theta_beta = theta_beta
#         self.theta_gamma = theta_gamma
#         self.num_iterations = num_iterations
#         self.spatial_ker_weights = None
#         self.bilateral_ker_weights = None
#         self.compatibility_matrix = None
#         super(CrfRnnLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Weights of the spatial kernel
       

#         super(CrfRnnLayer, self).build(input_shape)

#     def call(self, inputs):


#     def compute_output_shape(self, input_shape):
#         return input_shape