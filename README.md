# EMTOM

EMTOM is an embodied Theory of Mind benchmark built on top of Habitat and the PARTNR environment stack. It evaluates whether multi-agent systems can solve tasks that require asymmetric information, communication, and reasoning about what other agents know.

The benchmark follows a simple research pipeline:

1. Explore scenes and discover useful mechanics.
2. Generate candidate tasks from a blank template using scene-grounded context.
3. Verify solvability with static checks, PDDL checks, and golden-trajectory execution.
4. Judge whether the task genuinely requires Theory of Mind reasoning.
5. Benchmark agent models on the finalized task set.

The architecture and benchmark semantics are defined in [docs/benchmark-architecture.md](/data4/parth/Partnr-EmToM/docs/benchmark-architecture.md). The main operator entrypoint is [`./emtom/run_emtom.sh`](/data4/parth/Partnr-EmToM/emtom/run_emtom.sh).

## Repository Structure

- [`emtom/pddl`](/data4/parth/Partnr-EmToM/emtom/pddl) contains goal syntax, epistemic compilation, and solvability checks.
- [`emtom/task_gen`](/data4/parth/Partnr-EmToM/emtom/task_gen) contains task authoring, validation, calibration, and submission logic.
- [`emtom/runner`](/data4/parth/Partnr-EmToM/emtom/runner) contains runtime execution in Habitat.
- [`emtom/cli`](/data4/parth/Partnr-EmToM/emtom/cli) contains the user-facing command surface.
- [`docs`](/data4/parth/Partnr-EmToM/docs) contains the conceptual source of truth for benchmark behavior.

## Requirements and Setup

### System Requirements

- Linux environment with `conda` or `mamba`
- Python 3.9 environment for the main Habitat stack
- CUDA-capable GPU for Habitat-backed exploration, generation, `test-task`, and benchmarking
- Git LFS for large dataset assets

### Core Environment

Create the main environment expected by the repo:

```bash
conda create -n habitat-llm python=3.9.2 cmake=3.14.0 -y
conda activate habitat-llm
git submodule sync
git submodule update --init --recursive
```

Install the core dependencies:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
pip install -e ./third_party/habitat-lab/habitat-lab
pip install -e ./third_party/habitat-lab/habitat-baselines
pip install -e ./third_party/transformers-CFG
pip install -r requirements.txt
pip install -e .
```

If your system has library-linking issues, make sure the Conda environment libraries are on `LD_LIBRARY_PATH`.

### Data Dependencies

Download the simulator assets, HSSD scenes, PARTNR episodes, and checkpoints used by the benchmark:

```bash
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids --data-path data/ --no-replace --no-prune
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive

git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
cd data/versioned_data/hssd-hab && git lfs pull && cd ../../..
ln -s versioned_data/hssd-hab data/hssd-hab

git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
cd data/versioned_data/partnr_episodes && git lfs pull && cd ../../..
mkdir -p data/datasets
ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes
ln -s versioned_data/partnr_episodes/checkpoints data/models
```

### API Keys and External Tools

Create a repo-root `.env` file or export credentials in your shell for the providers you use:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

Task generation can launch an external SWE-agent CLI:

- `mini`: requires `mini-swe-agent` installed in the operator environment and `mini` on `PATH`
- `claude`: requires the relevant Claude CLI setup
- `codex`: requires the Codex CLI setup

Each task-generation run creates its own isolated workspace under `tmp/task_gen/<run_id>/.venv`, but the external agent executable itself must already be available in the main operator environment.

### What Requires a GPU

These commands require Habitat plus a GPU-backed environment:

- `explore`
- `generate`
- `new-scene`
- `test-task`
- `verify`
- `benchmark`
- `benchmark-suite`

These commands can be used for lightweight validation without Habitat-backed execution:

- `validate-task`
- `verify-static`
- `verify-pddl`
- `judge`

Synthetic fallback scenes may be used during lightweight authoring, but submitted benchmark tasks must still reference real dataset `scene_id` and `episode_id`.

## Running the Benchmark

All major operations go through [`./emtom/run_emtom.sh`](/data4/parth/Partnr-EmToM/emtom/run_emtom.sh).

### Explore

Run LLM-guided scene exploration:

```bash
./emtom/run_emtom.sh explore --steps 30 --model gpt-5.4
```

### Generate Tasks

Generate tasks with the external task-generation loop:

```bash
./emtom/run_emtom.sh generate \
  --task-gen-agent mini \
  --model gpt-5.2 \
  --target-model gpt-5.2 \
  --seed-tasks-dir data/emtom/tasks \
  --num-tasks 4
```

Current generation behavior:

- generation starts from a blank task template, not from a copied seed task
- sampled tasks are used only as inspiration
- calibration is tied to a `--target-model` and desired `--target-pass-rate`
- `test_task` evaluates both `standard` and `baseline`
- for competitive tasks, `baseline` runs as a two-phase solo-team check

Important generation flags:

- `--task-gen-agent`: external authoring agent, one of `mini`, `claude`, or `codex`
- `--target-model`: model whose calibration defines pass/fail seed buckets
- `--target-pass-rate`: desired pass rate for the target model
- `--seed-tasks-dir`: existing task pool used to construct `sampled_tasks/`
- `--seed-pass-ratio` and `--seed-fail-ratio`: logical pass/fail seed mixture
- `--k-level`: allowed ToM depth levels for generated tasks

### Verify and Judge Tasks

Static validation:

```bash
./emtom/run_emtom.sh validate-task --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh verify-static --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh verify-pddl --task data/emtom/tasks/my_task.json
```

Golden-trajectory execution and ToM judging:

```bash
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --model gpt-5.4
```

### Benchmark Models

Run the active benchmark over a task set:

```bash
./emtom/run_emtom.sh benchmark --tasks-dir data/emtom/tasks --model gpt-5.4
```

Useful benchmark variants:

```bash
./emtom/run_emtom.sh benchmark --task data/emtom/tasks/my_task.json --model gpt-5.4
./emtom/run_emtom.sh benchmark --tasks-dir data/emtom/tasks --model gpt-5.4 --run-mode baseline
./emtom/run_emtom.sh benchmark --tasks-dir data/emtom/tasks --model gpt-5.4 --observation-mode vision
./emtom/run_emtom.sh benchmark --tasks-dir data/emtom/tasks --model gpt-5.4 --max-workers 8
```

The benchmark distinguishes:

- `functional_success`: whether agents complete the task under the benchmark runtime
- `literal_tom_probe`: end-of-episode probe performance derived from `K()` goals

These metrics should be reported separately.

### Benchmark Suites and Campaigns

The shell entrypoint also exposes:

- `benchmark-suite`: benchmark one task set across multiple models
- `campaign`: manage the active benchmark campaign in `data/emtom/results/`

Use `./emtom/run_emtom.sh` with no arguments to see the full command surface and current flags.

### Bulk Generation

For parallel task generation across GPUs, use [`./emtom/bulk_generate.sh`](/data4/parth/Partnr-EmToM/emtom/bulk_generate.sh):

```bash
./emtom/bulk_generate.sh --num-tasks 8 --task-gen-agent mini --model gpt-5.2
```

Useful variants:

```bash
./emtom/bulk_generate.sh --dry-run --num-tasks 8 --task-gen-agent mini --model gpt-5.2
./emtom/bulk_generate.sh --per-gpu 1 --num-tasks 8 --task-gen-agent mini --model gpt-5.2
```

Bulk generation writes run manifests, worker status, traces, and logs under `outputs/generations/<run_id>/`.

## Notes for Researchers

- Use [`docs/benchmark-architecture.md`](/data4/parth/Partnr-EmToM/docs/benchmark-architecture.md) as the authoritative description of benchmark semantics.
- Keep exactly one active benchmark campaign in `data/emtom/results/`; archive incompatible campaigns before starting a new one.
- When benchmark architecture changes, update the docs in the same change.
