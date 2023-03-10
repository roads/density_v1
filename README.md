Author: Brett D. Roads

# Purpose 

To explore the role of exemplar density on human-perceived similarity.

# Usage

The repository is not a full-fledged package. You will need to ensure your python environment has `psiz` installed and execute scripts from within the directory.

NOTE: Scripts and directories using v08 do so to indicate the version of psiz needed/used.

# Experimental design and data.

The stimuli consist of abstract shapes.
The first phase (`phase0`) involves a norming study to determine a workable psychological space.
The second phase (`phase1`) involves experience manipulation to test density hypothesis.

Data source: https://github.com/lbokeria/norming_1d_objects_analysis/blob/main/results/preprocessed_data/triplet_task_long_form.csv

# Project structure

Use standard embedding structure.
* `create_catalog.py` Create a psiz `Catalog` object for all of the experimental stimuli. Call this script first.
* `data_adapter_v08.py` Adapt the raw csv data into a psiz-consumable `Observations` object. Call this script second.
* Analysis scripts
    * `phase0_fit_v08.py` Infer (and save) embedding models using norming data. The `phase_0` models do not include potential experience-mediated density interactions.
        * e.g., `python phase0_fit_v08.py --arch_id 0 --input_id 0`
    * `phase0_plot_v08.py` Visualize `phase_0` embeddings.
        * e.g., `python phase0_plot_v08.py --arch_id 2 --input_id 0 --run_id 1`

## Model identifiers

* Model names follow `emb_{arch_id}-{input_id}-{n_dim}-{run_id}`
* Architecture IDs `arch_id`
    * See `/assets/models/arch_id.csv`
* Input IDs `input_id`
    * See `/assets/models/input_id.csv`
* Dimensionality `n_dim`
    * Only using 1D space (i.e., n_dim=1)
* Run identifier `run_id`
    * Indicates restart run. Used three restarts for all architectures.

## Results

### arch_id=0, input_id=0
Using n_restart=3, the best validation loss is 0.29 (run_id=0).

### arch_id=1, input_id=0
Using n_restart=3, the best validation loss is 0.29, achieved with `beta=0.11` (run_id=2).

### arch_id=2, input_id=0
Using n_restart=3, the best validation loss is 0.29 (run_id=1).