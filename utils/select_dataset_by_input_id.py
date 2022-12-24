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
"""Module of utility functions.

Functions:
    select_dataset_by_input_id: Select samples based on `input_id`.

"""

import tensorflow as tf


def select_dataset_by_input_id(tfds_all, input_id):
    """Grab appropriate samples based on `input_id`."""
    if input_id == 0:
        # Include all real subjects (remove ideal responder,i.e., agent_id=0).
        def filter_fn(x, y, w):
            return tf.math.not_equal(x['agent_id'][0][0], 0)
        tfds_all = tfds_all.filter(filter_fn)
    elif input_id > 0 and input_id < 12:
        # Grab data with `agent_id` matching `input_id`.
        def filter_fn(x, y, w):
            return tf.math.equal(x['agent_id'][0][0], input_id)
        tfds_all = tfds_all.filter(filter_fn)
    elif input_id == 12:
        # Perfect responder only x1
        def filter_fn(x, y, w):
            return tf.math.equal(x['agent_id'][0][0], 0)
        tfds_all = tfds_all.filter(filter_fn)
    elif input_id == 13:
        # Perfect responder only, but x11 to mimic data from multiple agents.
        def filter_fn(x, y, w):
            return tf.math.equal(x['agent_id'][0][0], 0)
        tfds_all = tfds_all.filter(filter_fn)
        tfds_all = tfds_all.repeat(count=11)
    else:
        raise NotImplementedError('Invalid `input_id`.')
    return tfds_all
