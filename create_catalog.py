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

from pathlib import Path

import numpy as np
import pandas as pd
from psiz.catalog import Catalog


def main(fp_project):
    """Run script."""
    # Infer catalog for phase 0.
    catalog = infer_catalog_phase0(fp_project)
    fp_catalog = fp_project / Path('assets', 'stimuli', 'catalog_phase0.hdf5')
    catalog.save(fp_catalog)


def infer_catalog_phase0(fp_project):
    """Infer catalog for phase 0."""
    # Settings.
    fp_triplets_raw = fp_project / Path(
        'assets', 'raw', 'triplet_task_long_form_v1_phase0.csv'
    )

    # Read in csv data.
    df_obs = pd.read_csv(
        fp_triplets_raw
    )
    # Grab columns we need.
    df_obs = df_obs[[
        'query_stimulus', 'ref_left_stimulus', 'ref_right_stimulus',
        'qc_pass', 'trial_stage'
    ]]
    # Drop bad subjects.
    df_obs = df_obs[df_obs['qc_pass'] == 1]
    # Drop 'practice' trials and keep 'triplet_task' trials.
    df_obs = df_obs[df_obs['trial_stage'] == 'triplet_task']

    df_filenames = pd.concat(
        [
            df_obs['query_stimulus'], df_obs['ref_left_stimulus'],
            df_obs['ref_right_stimulus']
        ]
    )

    # Determine unique stimuli.
    unique_filenames = np.sort(pd.unique(df_filenames))
    n_stimuli = len(unique_filenames)

    stimulus_id = np.arange(n_stimuli, dtype=np.int32)

    return Catalog(stimulus_id, unique_filenames)


if __name__ == "__main__":
    fp_project = Path.home() / Path('projects', 'psiz-projects', 'density_v1')
    main(fp_project)
