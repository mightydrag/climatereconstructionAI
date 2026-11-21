# CRAI (Climate Reconstruction AI)

Software to train/evaluate models to reconstruct missing values in climate data (e.g., HadCRUT4) based on a U-Net with partial convolutions.

## Dependencies
- pytorch>=1.11.0
- tqdm>=4.64.0
- torchvision>=0.12.0
- torchmetrics>=0.11.2
- numpy>=1.21.6
- matplotlib>=3.5.1
- tensorboardX>=2.5
- tensorboard>=2.9.0
- xarray>=2022.3.0
- dask>=2022.7.0
- netcdf4>=1.5.8
- setuptools==59.5.0
- xesmf>=0.6.2
- cartopy>=0.20.2
- numba>=0.55.1

Use the python version 3.10.20

Install and use mamba following these instructions:
https://github.com/conda-forge/miniforge

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
mamba env create -f environment.yml
```
To activate the environment, use:
```bash
mamba activate crai
```

`environment-cuda.yml` should be used when working with GPUs using CUDA.

Add the libraries to the bashrc:
```
echo 'export LD_LIBRARY_PATH=/home/u/u244085/miniforge3/envs/crai/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

Additionally these packages are required:

- properscoring
- torch==2.12.1 torchvision==0.27.1
- spei
- wandb

## Installation

`climatereconstructionAI` can be installed using `pip` in the current directory:
```bash
pip install .
```

## Usage

The software can be used to:
- train a model (**training**)
- infill climate datasets using a trained model (**evaluation**)

### Input data
The directory containing the climate datasets should have the following sub-directories:
- `data` and `val` for training
- `test` for evaluation

The climate datasets should be in netCDF format and placed in the corresponding sub-directories.

The missing values can be defined separately as masks containing zeros (for the missing values) and ones (for the valid values). These masks should be in netCDF format and have the same dimension as the climate dataset. For the training, it is possible to shuffle the sequence of masks by using the "shuffle-masks" option.

A PyTorch model is required for the evaluation.

### Bias-correcting driving inputs with quantile delta mapping (QDM)
If a model trained on a reference dataset (e.g. CERRA) should be driven with another dataset (e.g. MPI-GE), the driving fields can first be mapped into the reference space with quantile delta mapping (QDM; Cannon, Sobie & Murdock 2015, *Journal of Climate* 28(17), 6938-6959, https://doi.org/10.1175/JCLI-D-14-00754.1). QDM corrects the distributional bias of the driving data against the reference over a calibration period while preserving the driving data's own changes (e.g. a scenario signal). Precipitation-like variables are corrected multiplicatively, other variables additively; optionally per calendar month.

Fit the transfer functions on the overlap period of reference and driving data (here: CERRA and the MPI-GE historical run, 1985-2014):
```bash
python -m climatereconstructionai.quantile_mapping fit \
  --ref cerra.nc --mod mpi-ge-historical.nc --mod-fut mpi-ge-scenario.nc \
  --vars tas,pr --hist-period 1985-2014 --fut-period 2071-2100 \
  --group-by-month --out qdm_cerra_mpige.npz
```
Then either pre-correct whole files:
```bash
python -m climatereconstructionai.quantile_mapping apply \
  --in mpi-ge-scenario.nc --out mpi-ge-scenario-qdm.nc --qdm qdm_cerra_mpige.npz
```
or let the data loader apply an (annually fitted) QDM on the fly when loading inputs, by passing `--qdm-input-file qdm_cerra_mpige.npz` to `crai-train`/`crai-evaluate`.

### Execution

Once installed, the package can be used as:
- a command line interface (CLI):
  - training:
  ```bash
  crai-train
  ```
  - evaluation:
  ```bash
  crai-evaluate
  ```
- a Python library:
  - training:
  ```python
  from climatereconstructionai import train
  train()
  ```
  - evaluation:
  ```python
  from climatereconstructionai import evaluate
  evaluate()
  ```

For more information about the arguments:
```bash
usage: crai-train [-h] [--data-root-dir DATA_ROOT_DIR] [--mask-dir MASK_DIR] [--log-dir LOG_DIR] [--data-names DATA_NAMES] [--mask-names MASK_NAMES] [--data-types DATA_TYPES]
                  [--n-target-data N_TARGET_DATA] [--device DEVICE] [--shuffle-masks] [--channel-steps CHANNEL_STEPS] [--lstm-steps LSTM_STEPS] [--gru-steps GRU_STEPS]
                  [--encoding-layers ENCODING_LAYERS] [--pooling-layers POOLING_LAYERS] [--conv-factor CONV_FACTOR] [--weights WEIGHTS] [--steady-masks STEADY_MASKS]
                  [--loop-random-seed LOOP_RANDOM_SEED] [--cuda-random-seed CUDA_RANDOM_SEED] [--deterministic] [--attention] [--channel-reduction-rate CHANNEL_REDUCTION_RATE] [--disable-skip-layers]
                  [--disable-first-bn] [--masked-bn] [--lazy-load] [--global-padding] [--gnn-layers GNN_LAYERS] [--gnn-neighbors GNN_NEIGHBORS] [--target-aggregation {none,sum,mean}] [--target-time-offset TARGET_TIME_OFFSET] [--normalize-data] [--n-filters N_FILTERS] [--out-channels OUT_CHANNELS] [--dataset-name DATASET_NAME]
                  [--min-bounds MIN_BOUNDS] [--max-bounds MAX_BOUNDS] [--profile] [--val-names VAL_NAMES] [--snapshot-dir SNAPSHOT_DIR] [--resume-iter RESUME_ITER] [--batch-size BATCH_SIZE]
                  [--n-threads N_THREADS] [--multi-gpus] [--finetune] [--lr LR] [--lr-finetune LR_FINETUNE] [--max-iter MAX_ITER] [--log-interval LOG_INTERVAL]
                  [--lr-scheduler-patience LR_SCHEDULER_PATIENCE] [--save-model-interval SAVE_MODEL_INTERVAL] [--n-final-models N_FINAL_MODELS] [--final-models-interval FINAL_MODELS_INTERVAL]
                  [--loss-criterion LOSS_CRITERION] [--eval-timesteps EVAL_TIMESTEPS] [-f LOAD_FROM_FILE] [--vlim VLIM] [--lambda-loss LAMBDA_LOSS] [--val-metrics VAL_METRICS]
                  [--tensor-plots TENSOR_PLOTS] [--early-stopping-delta EARLY_STOPPING_DELTA] [--early-stopping-patience EARLY_STOPPING_PATIENCE] [--n-iters-val N_ITERS_VAL]

options:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --log-dir LOG_DIR     Directory where the log files will be stored
  --data-names DATA_NAMES
                        Comma separated list of netCDF files (climate dataset) for training/infilling
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same order as data-names and mask-names
  --n-target-data N_TARGET_DATA
                        Number of data-names (from last) to be used as target data
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --shuffle-masks       Select mask indices randomly
  --channel-steps CHANNEL_STEPS
                        Comma separated number of considered sequences for channeled memory:past_steps,future_steps
  --lstm-steps LSTM_STEPS
                        Comma separated number of considered sequences for lstm: past_steps,future_steps
  --gru-steps GRU_STEPS
                        Comma separated number of considered sequences for gru: past_steps,future_steps
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --conv-factor CONV_FACTOR
                        Number of channels in the deepest layer
  --weights WEIGHTS     Initialization weight
  --steady-masks STEADY_MASKS
                        Comma separated list of netCDF files containing a single mask to be applied to all timesteps. The number of steady-masks must be the same as out-channels
  --loop-random-seed LOOP_RANDOM_SEED
                        Random seed for iteration loop
  --cuda-random-seed CUDA_RANDOM_SEED
                        Random seed for CUDA
  --deterministic       Disable cudnn backends for reproducibility
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --disable-first-bn    Disable the batch normalization on the first layer
  --masked-bn           Use masked batch normalization instead of standard BN
  --lazy-load           Use lazy loading for large datasets
  --global-padding      Use a custom padding for global dataset
  --gnn-layers GNN_LAYERS
                        Number of graph convolution (GNN) layers applied at the bottleneck of the U-Net, treating grid cells as nodes (0 disables the GNN)
  --gnn-neighbors GNN_NEIGHBORS
                        Number of grid neighbours per node used for message passing in the GNN layers (4 or 8)
  --target-aggregation {none,sum,mean}
                        Aggregate the model output over the input time window before the loss/evaluation ('sum' or 'mean'), for sub-daily inputs with an aggregated (e.g. daily) target. Requires --n-target-data > 0 and recurrent steps (--lstm-steps/--gru-steps). Default 'none' keeps the center-timestep behavior
  --target-time-offset TARGET_TIME_OFFSET
                        Offset (in input time steps) of each aggregated target sample within the input files, used with --target-aggregation. Example: 2 for 3-hourly inputs starting at 00 UTC when the daily target is a 06UTC-06UTC accumulation
  --precip-log-transform Apply log1p to precipitation data types before normalization (reduces skewness of the precipitation distribution)
  --precip-log-names PRECIP_LOG_NAMES
                          Comma separated data type names treated as precipitation for --precip-log-transform
  --normalize-data      Normalize the input climate data to 0 mean and 1 std
  --n-filters N_FILTERS
                        Number of filters for the first/last layer
  --out-channels OUT_CHANNELS
                        Number of channels for the output data
  --dataset-name DATASET_NAME
                        Name of the dataset for format checking
  --min-bounds MIN_BOUNDS
                        Comma separated list of values defining the permitted lower-bound of output values
  --max-bounds MAX_BOUNDS
                        Comma separated list of values defining the permitted upper-bound of output values
  --profile             Profile code using tensorboard profiler
  --val-names VAL_NAMES
                        Comma separated list of netCDF files (climate dataset) for validation
  --snapshot-dir SNAPSHOT_DIR
                        Parent directory of the training checkpoints and the snapshot images
  --resume-iter RESUME_ITER
                        Iteration step from which the training will be resumed
  --batch-size BATCH_SIZE
                        Batch size
  --n-threads N_THREADS
                        Number of workers used in the data loader
  --multi-gpus          Use multiple GPUs, if any
  --finetune            Enable the fine tuning mode (use fine tuning parameterization and disable batch normalization
  --lr LR               Learning rate
  --lr-finetune LR_FINETUNE
                        Learning rate for fine tuning
  --max-iter MAX_ITER   Maximum number of iterations
  --log-interval LOG_INTERVAL
                        Iteration step interval at which a tensorboard summary log should be written
  --lr-scheduler-patience LR_SCHEDULER_PATIENCE
                        Patience for the lr scheduler
  --save-model-interval SAVE_MODEL_INTERVAL
                        Iteration step interval at which the model should be saved
  --n-final-models N_FINAL_MODELS
                        Number of final models to be saved
  --final-models-interval FINAL_MODELS_INTERVAL
                        Iteration step interval at which the final models should be saved
  --loss-criterion LOSS_CRITERION
                        Index defining the loss function (0=original from Liu et al., 1=MAE of the hole region, 4=precipitation-weighted hole loss, 5/8=precipitation-weighted + Bernoulli, 9/10=two-stage Bernoulli-Gamma, 11=gamma; individual loss terms can also be set via --lambda-loss)
  --eval-timesteps EVAL_TIMESTEPS
                        Sample indices for which a snapshot is created at each iter defined by log-interval
  -f LOAD_FROM_FILE, --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
  --vlim VLIM           Comma separated list of vmin,vmax values for the color scale of the snapshot images
  --lambda-loss LAMBDA_LOSS
                        Comma separated list of lambda factors (key) followed by their corresponding values.Overrides the loss_criterion pre-setting
  --precip-loss-alpha PRECIP_LOSS_ALPHA
                        Weight boost factor for wet pixels in the precipitation loss (used with loss-criterion 4)
  --bern-pos-weight BERN_POS_WEIGHT
                        Weight for wet pixels in the Bernoulli (rain occurrence) loss
  --bern-threshold BERN_THRESHOLD
                        Threshold on the (normalized) target above which a pixel counts as wet in the Bernoulli loss
  --gamma-shape GAMMA_SHAPE
                        Shape parameter of the gamma distribution used in the gamma and Bernoulli-Gamma losses
  --gamma-threshold GAMMA_THRESHOLD
                        Threshold on the (normalized) target above which a pixel contributes to the gamma (amount) loss
  --gamma-data-types GAMMA_DATA_TYPES
                        Comma separated data type names to which the gamma/Bernoulli-Gamma loss is applied (default: all output channels)
  --val-metrics VAL_METRICS
                        Comma separated list of metrics that are evaluated on the val dataset at log-interval
  --tensor-plots TENSOR_PLOTS
                        Comma separated list of 2D plots to be added to tensorboard (error, distribution, correlation)
  --early-stopping-delta EARLY_STOPPING_DELTA
                        Mean relative delta of the val loss used for the termination criterion
  --early-stopping-patience EARLY_STOPPING_PATIENCE
                        Number of log-interval iterations used for the termination criterion
  --n-iters-val N_ITERS_VAL
                        Number of batch iterations used to average the validation loss
```

```bash
usage: crai-evaluate [-h] [--data-root-dir DATA_ROOT_DIR] [--mask-dir MASK_DIR] [--log-dir LOG_DIR] [--data-names DATA_NAMES] [--mask-names MASK_NAMES] [--data-types DATA_TYPES]
                     [--n-target-data N_TARGET_DATA] [--device DEVICE] [--shuffle-masks] [--channel-steps CHANNEL_STEPS] [--lstm-steps LSTM_STEPS] [--gru-steps GRU_STEPS]
                     [--encoding-layers ENCODING_LAYERS] [--pooling-layers POOLING_LAYERS] [--conv-factor CONV_FACTOR] [--weights WEIGHTS] [--steady-masks STEADY_MASKS]
                     [--loop-random-seed LOOP_RANDOM_SEED] [--cuda-random-seed CUDA_RANDOM_SEED] [--deterministic] [--attention] [--channel-reduction-rate CHANNEL_REDUCTION_RATE] [--disable-skip-layers]
                     [--disable-first-bn] [--masked-bn] [--lazy-load] [--global-padding] [--gnn-layers GNN_LAYERS] [--gnn-neighbors GNN_NEIGHBORS] [--target-aggregation {none,sum,mean}] [--target-time-offset TARGET_TIME_OFFSET] [--normalize-data] [--n-filters N_FILTERS] [--out-channels OUT_CHANNELS] [--dataset-name DATASET_NAME]
                     [--min-bounds MIN_BOUNDS] [--max-bounds MAX_BOUNDS] [--profile] [--model-dir MODEL_DIR] [--model-names MODEL_NAMES] [--evaluation-dirs EVALUATION_DIRS] [--eval-names EVAL_NAMES]
                     [--use-train-stats] [--create-graph] [--plot-results PLOT_RESULTS] [--partitions PARTITIONS] [--maxmem MAXMEM] [--split-outputs] [-f LOAD_FROM_FILE]

options:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --log-dir LOG_DIR     Directory where the log files will be stored
  --data-names DATA_NAMES
                        Comma separated list of netCDF files (climate dataset) for training/infilling
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same order as data-names and mask-names
  --n-target-data N_TARGET_DATA
                        Number of data-names (from last) to be used as target data
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --shuffle-masks       Select mask indices randomly
  --channel-steps CHANNEL_STEPS
                        Comma separated number of considered sequences for channeled memory:past_steps,future_steps
  --lstm-steps LSTM_STEPS
                        Comma separated number of considered sequences for lstm: past_steps,future_steps
  --gru-steps GRU_STEPS
                        Comma separated number of considered sequences for gru: past_steps,future_steps
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --conv-factor CONV_FACTOR
                        Number of channels in the deepest layer
  --weights WEIGHTS     Initialization weight
  --steady-masks STEADY_MASKS
                        Comma separated list of netCDF files containing a single mask to be applied to all timesteps. The number of steady-masks must be the same as out-channels
  --loop-random-seed LOOP_RANDOM_SEED
                        Random seed for iteration loop
  --cuda-random-seed CUDA_RANDOM_SEED
                        Random seed for CUDA
  --deterministic       Disable cudnn backends for reproducibility
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --disable-first-bn    Disable the batch normalization on the first layer
  --masked-bn           Use masked batch normalization instead of standard BN
  --lazy-load           Use lazy loading for large datasets
  --global-padding      Use a custom padding for global dataset
  --gnn-layers GNN_LAYERS
                        Number of graph convolution (GNN) layers applied at the bottleneck of the U-Net, treating grid cells as nodes (0 disables the GNN)
  --gnn-neighbors GNN_NEIGHBORS
                        Number of grid neighbours per node used for message passing in the GNN layers (4 or 8)
  --normalize-data      Normalize the input climate data to 0 mean and 1 std
  --n-filters N_FILTERS
                        Number of filters for the first/last layer
  --out-channels OUT_CHANNELS
                        Number of channels for the output data
  --dataset-name DATASET_NAME
                        Name of the dataset for format checking
  --min-bounds MIN_BOUNDS
                        Comma separated list of values defining the permitted lower-bound of output values
  --max-bounds MAX_BOUNDS
                        Comma separated list of values defining the permitted upper-bound of output values
  --profile             Profile code using tensorboard profiler
  --model-dir MODEL_DIR
                        Directory of the trained models
  --model-names MODEL_NAMES
                        Model names. If 'best', 'latest' or 'final' is given but no such file exists in model-dir, the checkpoint with the highest iteration number is used
  --evaluation-dirs EVALUATION_DIRS
                        Directory where the output files will be stored
  --eval-names EVAL_NAMES
                        Prefix used for the output filenames
  --use-train-stats     Use mean and std from training data for normalization
  --create-graph        Create a Tensorboard graph of the NN
  --plot-results PLOT_RESULTS
                        Create plot images of the results for the comma separated list of time indices
  --partitions PARTITIONS
                        Split the climate dataset into several partitions along the time coordinate
  --maxmem MAXMEM       Maximum available memory in MB (overwrite partitions parameter)
  --split-outputs       Do not merge the outputs when using multiple models and/or partitions
  -f LOAD_FROM_FILE, --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
```

## Distribution-aware losses, GNN and temporal architectures

Besides the reconstruction losses above, the training supports distribution-aware loss terms that improve the representation of variance and tail behaviour (e.g. of sub-daily precipitation, wind or humidity):

- `--loss-criterion 9` / `10`: two-stage **Bernoulli-Gamma** loss (Bernoulli negative log-likelihood for the occurrence plus a gamma negative log-likelihood for the amount on wet pixels), on the hole or the valid region. The occurrence threshold and wet-pixel weight are set via `--bern-threshold` / `--bern-pos-weight`; the gamma shape via `--gamma-shape`.
- `--loss-criterion 11`: **gamma** loss (negative log-likelihood of a gamma distribution) on the hole region, e.g. for wind speed or humidity.
- Individual loss terms can also be combined freely via `--lambda-loss`, e.g. `--lambda-loss valid,1.0 bern_gamma,6.0 tv,0.1`. The available terms are `valid`, `hole`, `tv`, `tv_valid`, `precip`, `precip_valid`, `bern`, `bern_valid`, `gamma`, `gamma_valid`, `bern_gamma`, `bern_gamma_valid`, `prc` and `style`.
- With `--gamma-data-types pr,hurs` the gamma/Bernoulli-Gamma terms are restricted to the output channels of the listed data types (useful for multi-variable setups where, e.g., only precipitation and humidity should be treated distributionally).
- Combined with `--precip-log-transform`, a threshold of `0.0` in the transformed space separates dry from wet pixels.

For the architecture:

- `--gnn-layers N` adds N graph convolution layers at the bottleneck of the U-Net, where each grid cell is a node and messages are passed between its `--gnn-neighbors` (4 or 8) nearest grid neighbours (circular in longitude with `--global-padding`). This provides GNN-style spatial message passing without additional dependencies.
- Temporal dependence between time steps is modelled with the recurrent options `--lstm-steps past,future` (ConvLSTM) or `--gru-steps past,future` (TrajGRU), or with `--channel-steps past,future` for channeled memory; these can be combined with the GNN layers.
- `--target-aggregation sum|mean` enables **sub-daily inputs with an aggregated (e.g. daily) target**: the last `--n-target-data` files have the target (e.g. daily) time axis; input files may be sub-daily (3-hourly, hourly, ...) or daily themselves (e.g. daily precipitation as an additional predictor). Each sample is indexed in target space, sub-daily inputs are loaded with the window (`--lstm-steps past,future`) around `sample_index * steps_per_day + --target-time-offset` (e.g. `--lstm-steps 0,7` for 3-hourly inputs covering exactly one day; `--target-time-offset 2` if the daily target is a 06UTC-06UTC accumulation but the 3-hourly files start at 00 UTC), daily inputs are broadcast over the window, and the model output is aggregated over the window (`sum` for precipitation totals, `mean` for the other variables) before the loss and at evaluation. Pass the same flags to `crai-evaluate`. Note: if the sub-daily inputs are only interpolations of daily data, they carry no additional information and this will not improve results.

Example (sub-daily precipitation infilling with a Bernoulli-Gamma loss, GNN bottleneck and temporal recurrence):

```bash
crai-train --data-names pr.nc --data-types pr --n-target-data 1 --out-channels 1 \
  --precip-log-transform --normalize-data \
  --loss-criterion 9 --gamma-data-types pr \
  --gnn-layers 2 --gnn-neighbors 8 --lstm-steps 2,1
```

Example (genuinely sub-daily inputs downscaled to a daily precipitation target):

```bash
crai-train --data-names tas_3h.nc,pr_day.nc --data-types tas,pr \
  --n-target-data 1 --out-channels 1 \
  --precip-log-transform --normalize-data \
  --loss-criterion 9 \
  --lstm-steps 3,4 --target-aggregation sum \
  --gnn-layers 2 --gnn-neighbors 8
```

## Example

An example can be found in the directory `demo`.
The instructions to run the example are given in the README.md file.

## License

`CRAI` is licensed under the terms of the BSD 3-Clause license.

## Contributions

`CRAI` is maintained by the Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum).
- Previous contributing authors: Naoto Inoue, Christopher Kadow, Stephan Seitz
- Current contributing authors: Johannes Meuer, Maximilian Witte, Étienne Plésiat.
