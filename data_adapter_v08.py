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

import os
from pathlib import Path

import numpy as np
import pandas as pd
from psiz.catalog import load_catalog
import psiz.data


def main(fp_project):
    """Run script."""
    # Handle phase 0.
    fp_catalog = fp_project / Path('assets', 'stimuli', 'catalog_phase0.hdf5')
    ds_phase0 = data_adapter_phase0(fp_project, fp_catalog)

    # print('Total trials retained: {0}'.format(obs.n_trial))
    fp_ds = fp_project / Path('assets', 'data_v08', 'ds_phase0')
    ds_phase0.save(str(fp_ds))


def data_adapter_phase0(fp_project, fp_catalog):
    """Parse data for phase 0."""
    # Settings.
    fp_triplets_raw = fp_project / Path(
        'assets', 'raw', 'triplet_task_long_form.csv'
    )
    drop_list = [
        'sub008',
        'sub012'
    ]
    max_n_reference = 2
    n_select_hardcoded = 1

    catalog = load_catalog(fp_catalog)

    # Create name to idx map.
    name_idx_map = create_name_index_map_phase0(catalog)

    # Read in csv data.
    df_obs = pd.read_csv(fp_triplets_raw)
    # Grab columns we need.
    df_obs = df_obs[
        [
            'prolific_id',
            'trial_index',
            'query_item',
            'ref_left',
            'ref_right',
            'chosen_ref_value',
            'rt',
            'qc_pass',
            'trial_stage'
        ]
    ]

    # Drop bad subjects.
    df_obs = df_obs[df_obs['qc_pass'] == 1]

    # Drop 'practice' trials and keep 'triplet_task' trials.
    df_obs = df_obs[df_obs['trial_stage'] == 'triplet_task']

    # Infer mapping between identifier and anonymous "agent ID".
    # subject_id_unique = pd.unique(df_obs['prolific_id'].values)
    # Hardcode agent mapping.
    subject_id_unique = np.array([
        'perfect_responder',
        'sub001',
        'sub002',
        'sub003',
        'sub005',
        'sub006',
        'sub007',
        'sub009',
        'sub010',
        'sub011',
        'sub013',
        'sub014',
    ])

    agent_mapping = {}
    for idx, i_subj_id in enumerate(subject_id_unique):
        agent_mapping[i_subj_id] = idx

    # Drop subjects in drop list.
    for i_drop in drop_list:
        loc_keep = df_obs['prolific_id'] != i_drop
        df_obs = df_obs[loc_keep]
    len_before = len(df_obs)

    # Subject info.
    print('Retained subjects: {0}'.format(len(subject_id_unique)))

    # Drop trials with no choice data.
    loc_keep = df_obs['chosen_ref_value'].notnull()
    df_obs = df_obs[loc_keep]
    len_after = len(df_obs)
    n_drop = len_before - len_after
    print('Empty trials dropped: {0}'.format(n_drop))

    # Check for duplicates.
    df_new = df_obs.drop_duplicates()
    if len(df_obs) != len(df_new):
        print(
            'WARNING: Detected duplicate observations. Dropped duplicate data.'
        )
        df_obs = df_new
        del df_new
    n_trial = df_obs.shape[0]

    stim_id_unique, stim_id_counts = np.unique(
        np.hstack([
            df_obs['query_item'].values, df_obs['ref_left'].values,
            df_obs['ref_right'].values
        ]), return_counts=True
    )

    # Pre-allocate.
    stimulus_set = -1 * np.ones([n_trial, max_n_reference + 1], dtype=np.int32)
    agent_id = np.zeros([n_trial, 1], dtype=np.int32)
    outcome_idx = np.zeros([n_trial, 1], dtype=np.int32)
    rt_ms = np.zeros([n_trial, 1])
    for i_trial in range(n_trial):
        # Infer correct ordering of observed trial.
        # TODO don't need to reorder for psiz v0.8
        q = name_idx_map[df_obs.query_item.values[i_trial]]
        ref0 = name_idx_map[df_obs.ref_left.values[i_trial]]
        ref1 = name_idx_map[df_obs.ref_right.values[i_trial]]
        chosen_id = name_idx_map[int(df_obs.chosen_ref_value.values[i_trial])]

        stimulus_set[i_trial, 0] = q
        if chosen_id == ref0:
            stimulus_set[i_trial, 1] = ref0
            stimulus_set[i_trial, 2] = ref1
        elif chosen_id == ref1:
            stimulus_set[i_trial, 1] = ref1
            stimulus_set[i_trial, 2] = ref0
        else:
            raise ValueError(
                'Chosen index does not match available reference indices.'
            )
        agent_id[i_trial, 0] = agent_mapping[
            df_obs.prolific_id.values[i_trial]
        ]

        # Behavior.
        outcome_idx[i_trial, 0] = 0  # TODO
        rt_ms[i_trial, 0] = df_obs.rt.values[i_trial]

    # Create dataset.
    content = psiz.data.Rank(stimulus_set + 1, n_select=n_select_hardcoded)
    agent_id = psiz.data.Group(agent_id, name='agent_id')
    # name='2rank1/outcome'
    outcome = psiz.data.SparseCategorical(outcome_idx, depth=2)
    ds_phase0 = psiz.data.Dataset([content, agent_id, outcome])

    return ds_phase0.export(export_format='tfds')


def create_name_index_map_phase0(catalog):
    """Create name to idx map."""
    filename_list = catalog.file_path()
    stimulus_id_list = catalog.id()
    name_idx_map = {}
    for stim_id, fn in zip(stimulus_id_list, filename_list):
        fn_int = int(os.fspath(fn).split('/')[-1].split('F0Level')[1].split(
            'F1Level'
        )[0])
        name_idx_map[fn_int] = stim_id

    return name_idx_map


if __name__ == "__main__":
    fp_project = Path.home() / Path('projects', 'psiz-projects', 'density_v1')
    main(fp_project)
