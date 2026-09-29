# CQL HalfCheetah Training Evidence

## Experiment

- Algorithm: Conservative Q-Learning (CQL)
- Task: `mujoco/halfcheetah/medium-v0`
- Environment: `HalfCheetah-v5`
- Training steps: `100,000`
- Seeds: `0, 1, 2, 3, 4`
- Repository branch: `research-option-b`
- Training repository commit recorded in the archive:
  `0585df5f310ef6f5db82f892f32c4c4b2027f4ca`

## Training configuration

The archived training metadata reports:

- `StandardObservationScaler`
- `MinMaxActionScaler`
- `initial_alpha = 1.0`
- `alpha_learning_rate = 1e-4`

GPU assignment recorded in the training metadata:

- seed 0 -> GPU 0
- seed 1 -> GPU 1
- seed 2 -> GPU 0
- seed 3 -> GPU 1
- seed 4 -> GPU 0

## Training completion

All five requested seeds completed successfully.

- seed 0: COMPLETE, return code 0
- seed 1: COMPLETE, return code 0
- seed 2: COMPLETE, return code 0
- seed 3: COMPLETE, return code 0
- seed 4: COMPLETE, return code 0

The archived checkpoints are:

- seed 0:
  `cql_mujoco_halfcheetah_medium-v0_seed0.d3`
- seed 1:
  `cql_mujoco_halfcheetah_medium-v0_seed1.d3`
- seed 2:
  `cql_mujoco_halfcheetah_medium-v0_seed2.d3`
- seed 3:
  `cql_mujoco_halfcheetah_medium-v0_seed3.d3`
- seed 4:
  `cql_mujoco_halfcheetah_medium-v0_seed4.d3`

## Final training archive

Filename:

`CQL_HalfCheetah_FINAL_5_SEEDS.tar.gz`

SHA-256:

`c19148fe8e94eff435cdc9ff5093029570fde040b29d9255a4570cbb19b1eb63`

## Checkpoint SHA-256

### Seed 0

`917d6fc88993fcdd947d394fa9d8e081fa502161b983453ffd1276112cfbc7d4`

### Seed 1

`3947dd0e5f1ee7ed65b55c80983c8f1d61e159a126e4122456ca6429063bd6a9`

### Seed 2

`e6a796fe68bd0b26351b6c31d2d23e2765ced80c064d973e6368304f88ff7b9c`

### Seed 3

`a7fe574b3893700e6c72d76c7b2177e0646b52b406ae3f353c46b7cd40677905`

### Seed 4

`fe0b9a68f70d9729624e28e58e73be9371ab94034fbfeb895a968fd58ce1343a`

## Provenance note

The original archived status JSON files preserve their original contents unchanged.

Those status files contain the experiment label:

`IQL-100K-multi-seed-replication`

while their task and algorithm fields identify:

- task: `mujoco/halfcheetah/medium-v0`
- algorithm: `cql`

The final training metadata independently identifies this run as CQL HalfCheetah training. The original archived files are therefore preserved unchanged, and the label discrepancy is documented here rather than altering the raw training artifact.

## Verification

The final archive contains:

- five CQL HalfCheetah checkpoints
- five seed status files
- training logs
- per-stage SHA-256 manifests
- final SHA-256 manifest
- training metadata

All five seed status files report `COMPLETE` with return code `0`.

## Next phase

The verified training checkpoints will be used for the frozen CQL HalfCheetah consequence experiment.

No additional training is required for this cell.
