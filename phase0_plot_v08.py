# -*- coding: utf-8 -*-
# Copyright 2022 Brett D. Roads. All Rights Reserved.
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
"""Analyze non-density embedding model.

Example CPU usage:
`python phase0b.py --input_id 0`

Example GPU usage:
`python phase0b.py --input_id 0 --gpu 0`


"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psiz
import scipy.stats as st
from sklearn.linear_model import LinearRegression
import tensorflow as tf

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def analyze_model(fp_project, arch_id, input_id, run_id):
    """Run script."""
    # Settings.
    n_dim = 1
    fp_catalog = fp_project / Path('assets', 'stimuli', 'catalog_phase0.hdf5')

    model_name = 'emb_{0}-{1}-{2}-{3}'.format(
        arch_id, input_id, n_dim, run_id
    )
    fp_model = fp_project / Path('assets', 'models_v08', model_name)
    fp_fig = fp_project / Path(
        'results_v08', '{0}.tiff'.format(model_name)
    )

    # Plot settings.
    apply_plt_settings()

    # Load data.
    catalog = psiz.catalog.load_catalog(fp_catalog)

    # Get pixel values from filenames.
    fp_list = catalog.filepath()
    gen_space_arr = []
    for i_fp in fp_list:
        fn_float = float(os.fspath(i_fp).split('/')[-1].split(
            'F0Level'
        )[1].split('F1Level')[0])
        gen_space_arr.append(fn_float)
    gen_space_arr = np.array(gen_space_arr)

    # Load VI model.
    model = tf.keras.models.load_model(fp_model)

    # Compile settings.
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    print(
        '    beta: {0:.2f}'.format(
            model.behavior.kernel.similarity.beta.numpy()
        )
    )

    # Create visual.
    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    draw_figure(fig, model, catalog, n_dim, gen_space_arr)
    plt.savefig(
        os.fspath(fp_fig), format='tiff', bbox_inches="tight", dpi=300
    )


def draw_figure(fig, model, catalog, n_dim, gen_space_arr):
    """Draw figure."""
    # Settings
    gs = fig.add_gridspec(1, 1)

    # Plot embeddings.
    ax = fig.add_subplot(gs[0, 0])

    # Determine embedding limits.
    dist = model.behavior.percept.embeddings
    loc, cov = unpack_mvn(dist)
    if model.behavior.percept.mask_zero:
        # Drop placeholder stimulus.
        loc = loc[1:]
        cov = cov[1:]

    z_max = 1.3 * np.max(np.abs(loc))
    z_limits = [-z_max, z_max]

    if n_dim == 1:
        loc = np.squeeze(loc)

        # If necessary, flip model based on what should be the smallest value.
        if loc[2] > 0:
            loc = - loc

        # y_constant = np.zeros_like(loc)
        # y_constant = z_max * np.linspace(0., 1., num=len(loc))
        y_constant = gen_space_arr

        # Draw 1D HDI intervals.
        p = .95
        outside_prob = 1 - p
        r = st.norm.ppf(1 - outside_prob / 2)
        ci_half = []
        for i_cov in cov:
            ci_half.append(r * np.sqrt(i_cov[0, 0]))
        ci_half = np.stack(ci_half)
        # ax.errorbar(
        #     loc, y_constant, ecolor=exemplar_color_array, xerr=ci_half,
        #     linestyle=''
        # )
        ax.errorbar(
            loc, y_constant, ecolor='k', xerr=ci_half,
            linestyle=''
        )

        # Draw modes.
        # ax.scatter(loc, y_constant, c=exemplar_color_array)
        ax.scatter(loc, y_constant, c='k')

        # Draw linear regression.
        idx_sorted = np.argsort(loc)
        loc_sorted = loc[idx_sorted]
        y_sorted = y_constant[idx_sorted]

        loc_correction = np.mean(loc_sorted)
        loc_centered = loc_sorted - loc_correction
        loc_centered = np.reshape(loc_centered, [len(loc_centered), 1])
        y_correction = np.mean(y_sorted)
        y_centered = y_sorted - y_correction
        reg = LinearRegression(fit_intercept=False).fit(
            loc_centered, y_centered
        )
        loc_endpoints = loc_centered[[0, -1]] * 1.2
        linear_pred = reg.predict(loc_endpoints)
        ax.plot(
            loc_endpoints[:, 0] + loc_correction, linear_pred + y_correction,
            color='r', linestyle='--'
        )

        y_min = np.min(y_constant)
        y_max = np.max(y_constant)
        pad = .1 * (y_max - y_min)
        y_limits = [y_min - pad, y_max + pad]

        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel('Psychological Space')
        ax.set_ylabel('Generative Space')

    else:
        raise NotImplementedError

    ax.set_xlim(z_limits)
    ax.set_ylim(y_limits)
    ax.set_title('Embeddings (95% HDI)')

    gs.tight_layout(fig)


def unpack_mvn(dist):
    """Unpack multivariate normal distribution."""
    def diag_to_full_cov(v):
        """Convert diagonal variance to full covariance matrix.

        Assumes `v` represents diagonal variance elements only.
        """
        n_stimuli = v.shape[0]
        n_dim = v.shape[1]
        cov = np.zeros([n_stimuli, n_dim, n_dim])
        for i_stimulus in range(n_stimuli):
            cov[i_stimulus] = np.eye(n_dim) * v[i_stimulus]
        return cov

    loc = dist.mean().numpy()
    v = dist.variance().numpy()

    # Convert to full covariance matrix.
    cov = diag_to_full_cov(v)

    return loc, cov


def apply_plt_settings():
    small_size = 6
    medium_size = 8
    large_size = 10
    plt.rc('font', size=small_size)
    plt.rc('axes', titlesize=medium_size)
    plt.rc('axes', labelsize=small_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=small_size)
    plt.rc('figure', titlesize=large_size)


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
        '--run_id', type=int, default=0,
        help='Integer indicating the run ID. Will default to `0`'
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

    analyze_model(fp_project, args.arch_id, args.input_id, args.run_id)
