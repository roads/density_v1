# -*- coding: utf-8 -*-
# Copyright 2021 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Infer non-density embedding model with multiple restarts.

Example CPU usage:
`python phase0a_fit.py --input_id 0`

Example GPU usage:
`python phase0a_fit.py --input_id 0 --gpu 0`

"""

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import psiz
import tensorflow as tf
import tensorflow_probability as tfp

from utils.select_dataset_by_input_id import select_dataset_by_input_id

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class StochasticBehaviorModel(psiz.keras.StochasticModel):
    """A behavior model.

    No Gates.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize."""
        super(StochasticBehaviorModel, self).__init__(**kwargs)
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)


def infer_model(fp_project, arch_id, input_id):
    """Run script."""
    # Settings.
    n_restart = 3
    n_dim = 1

    epochs = 10000
    batch_size = 128
    fp_catalog = fp_project / Path('assets', 'stimuli', 'catalog_phase0.hdf5')
    fp_ds = fp_project / Path('assets', 'data_v08', 'ds_phase0')

    # Load data.
    catalog = psiz.catalog.load_catalog(fp_catalog)
    tfds_all = tf.data.Dataset.load(str(fp_ds))
    tfds_all = select_dataset_by_input_id(tfds_all, input_id)

    # Count trials.
    n_trial = 0
    for _ in tfds_all:
        n_trial = n_trial + 1

    # Partition data into 90% train, 10% validation set.
    n_trial_train = int(np.round(0.9 * n_trial))
    tfds_train = tfds_all.take(n_trial_train).cache().shuffle(
        buffer_size=n_trial_train, reshuffle_each_iteration=True
    ).batch(
        batch_size, drop_remainder=False
    )
    tfds_val = tfds_all.skip(n_trial_train).cache().batch(
        batch_size, drop_remainder=False
    )

    model_list = []
    result_list = []
    for i_restart in range(n_restart):
        model_name = 'emb_{0}-{1}-{2}-{3}'.format(
            arch_id, input_id, n_dim, i_restart
        )
        fp_model = fp_project / Path('assets', 'models_v08', model_name)
        fp_board = fp_project / Path('assets', 'logs_v08', 'fit', model_name)

        # Directory preparation.
        fp_board.mkdir(parents=True, exist_ok=True)
        # Remove existing TensorBoard logs.
        if fp_board.exists():
            shutil.rmtree(fp_board)

        # Build VI model.
        n_stimuli = 11
        if arch_id == 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
            model = build_model_0(n_stimuli, n_dim, n_trial_train, optimizer)
        elif arch_id == 1:
            optimizer = tf.keras.optimizers.Adam(learning_rate=.1)
            model = build_model_1(
                n_stimuli, n_dim, n_trial_train, optimizer, catalog
            )
        elif arch_id == 2:
            optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
            model = build_model_2(
                n_stimuli, n_dim, n_trial_train, optimizer, catalog
            )
        else:
            raise NotImplementedError

        # Define callbacks.
        fp_board_restart = fp_board / Path('restart_{0}'.format(i_restart))
        cb_board = tf.keras.callbacks.TensorBoard(
            log_dir=fp_board_restart,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None
        )
        cb_early = tf.keras.callbacks.EarlyStopping(
            'loss',
            patience=100,
            mode='min',
            restore_best_weights=False,
            verbose=1
        )
        callbacks = [cb_board, cb_early]

        # Infer embedding.
        history = model.fit(
            x=tfds_train, epochs=epochs, callbacks=callbacks, verbose=0
        )
        model_list.append(model)
        model.save(fp_model)

        val_metrics = model.evaluate(tfds_val, verbose=0, return_dict=True)

        result = {
            'train_loss': history.history['loss'][-1],
            'train_cce': history.history['cce'][-1],
            'val_loss': val_metrics['loss'],
            'val_cce': val_metrics['cce'],
        }
        result_list.append(result)

    best_loss = np.inf
    best_idx = -1
    for result_idx, i_result in enumerate(result_list):
        if i_result['val_loss'] < best_loss:
            best_loss = i_result['val_loss']
            best_idx = result_idx

    print('Best restart: {0}'.format(best_idx))
    result = result_list[best_idx]
    print(
        '    train_loss: {0:.2f} | train_cce: {1:.2f} | '.format(
            result['train_loss'], result['train_cce']
        )
    )
    print(
        '    val_loss: {0:.2f} | val_cce: {1:.2f} | '.format(
            result['val_loss'], result['val_cce']
        )
    )
    print('    beta: {0:.2f}'.format(
            model.behavior.kernel.similarity.beta.numpy()
        )
    )
    print(
        '    mean scale: {0:.5f}'.format(
            np.sqrt(
                np.mean(
                    model.behavior.percept.embeddings.variance().numpy()
                )
            )
        )
    )


def build_model_0(n_stimuli, n_dim, n_sample_train, optimizer):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli (not
            including placeholder).
        n_dim: Integer indicating the dimensionality of the embedding.
        n_sample_train: Integer indicating the number of training
            observations. Used to determine KL weight for variational
            inference.

    Returns:
        model: A TensorFlow Keras model.

    """
    kl_weight = 1. / n_sample_train
    beta_init = 10.
    prior_scale = .2  # pixel_std

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        (n_stimuli + 1),
        n_dim,
        mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        ),
        loc_trainable=True,
        scale_trainable=True
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
            scale_trainable=True
        )
    )
    percept = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior,
        prior=embedding_prior,
        kl_weight=kl_weight,
        kl_n_sample=30
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False,
            fit_gamma=False,
            fit_beta=False,
            beta_initializer=tf.keras.initializers.Constant(beta_init),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    behavior = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel
    )
    model = StochasticBehaviorModel(behavior=behavior, n_sample=30)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        weighted_metrics=[
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    )
    return model


def build_model_1(n_stimuli, n_dim, n_sample_train, optimizer, catalog):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli (not
            including placeholder).
        n_dim: Integer indicating the dimensionality of the embedding.
        n_sample_train: Integer indicating the number of training
            observations. Used to determine KL weight for variational
            inference.
        catalog:

    Returns:
        model: A TensorFlow Keras model.

    """
    # Settings
    kl_weight = 1. / n_sample_train
    beta_init = 1.
    prior_scale = 25.0
    posterior_scale = 5.0  # NOTE: Made separate value to speed up inference.
    loc_fixed = generative_coordinates(catalog, mask_zero=True, centered=True)

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(loc_fixed),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(posterior_scale).numpy()
        ),
        loc_trainable=False,
        scale_trainable=True
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
            scale_trainable=True
        )
    )
    percept = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior,
        prior=embedding_prior,
        kl_weight=kl_weight,
        kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False,
            fit_gamma=False,
            fit_beta=True,
            beta_initializer=tf.keras.initializers.Constant(beta_init),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    behavior = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel
    )
    model = StochasticBehaviorModel(behavior=behavior, n_sample=30)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        weighted_metrics=[
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    )
    return model


def build_model_2(n_stimuli, n_dim, n_sample_train, optimizer, catalog):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli (not
            including placeholder).
        n_dim: Integer indicating the dimensionality of the embedding.
        n_sample_train: Integer indicating the number of training
            observations. Used to determine KL weight for variational
            inference.
        catalog:

    Returns:
        model: A TensorFlow Keras model.

    """
    # Settings.
    kl_weight = 1. / n_sample_train
    beta_init = 0.11  # NOTE: Value determined from result of arch_id=1 fits.
    prior_scale = 25.0  # NOTE: Had to increase because of low beta.
    posterior_scale = 5.0  # NOTE: Made separate value to speed up inference.
    loc_init = generative_coordinates(catalog, mask_zero=True, centered=True)

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(loc_init),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(posterior_scale).numpy()
        ),
        loc_trainable=True,
        scale_trainable=True
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
            scale_trainable=True
        )
    )
    percept = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior,
        prior=embedding_prior,
        kl_weight=kl_weight,
        kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False,
            fit_gamma=False,
            fit_beta=False,
            beta_initializer=tf.keras.initializers.Constant(beta_init),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    behavior = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel
    )
    model = StochasticBehaviorModel(behavior=behavior, n_sample=30)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        weighted_metrics=[
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    )
    return model


def generative_coordinates(catalog, mask_zero=True, centered=False):
    # Get pixel values from filenames.
    fp_list = catalog.filepath()
    gen_space_arr = []
    for i_fp in fp_list:
        fn_float = float(os.fspath(i_fp).split('/')[-1].split(
            'F0Level'
        )[1].split('F1Level')[0])
        gen_space_arr.append(fn_float)
    gen_space_arr = np.array(gen_space_arr)

    # Create fixed coordinate locations. Subtract mean so coordinates are
    # centered.
    if centered:
        gen_space_arr = gen_space_arr - np.mean(gen_space_arr)

    if mask_zero:
        loc = np.hstack(
            (np.array([0]), gen_space_arr)
        )
    return loc


if __name__ == "__main__":
    fp_project = Path.home() / Path('projects', 'psiz-projects', 'density_v1')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch_id', type=int, default=0,
        help='Integer indicating the architecture ID. Will default to `0`'
    )
    parser.add_argument(
        '--input_id', type=int, default=0,
        help='Integer indicating the input ID. Will default to `0`'
    )
    parser.add_argument(
        '--gpu', type=int, default=-1,
        help='Integer indicating GPU to use. Will default to CPU only.'
    )
    args = parser.parse_args()
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    infer_model(fp_project, args.arch_id, args.input_id)
