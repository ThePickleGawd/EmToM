# EMTOM Evolve (Simple Mode)

`evolve` upgrades your existing task pool in parallel.

Default behavior is intentionally simple:
- seed from `data/emtom/tasks`
- generate upgrades in parallel
- keep pushing difficulty until each ladder tier reaches the target pass-rate
- guide generation toward higher-order ToM mix by default

## Quick Start

```bash
./emtom/run_emtom.sh evolve
```

This is the recommended default call.

## Common Calls

Harder tasks (general):

```bash
./emtom/run_emtom.sh evolve --focus difficulty
```

Higher-order ToM emphasis:

```bash
./emtom/run_emtom.sh evolve --focus tom
```

Use more parallel workers:

```bash
./emtom/run_emtom.sh evolve --max-workers 24
```

Use all categories (balanced round-robin across workers):

```bash
./emtom/run_emtom.sh evolve --category all
```

Use a subset of categories:

```bash
./emtom/run_emtom.sh evolve --category cooperative,mixed
```

Resume an interrupted run:

```bash
./emtom/run_emtom.sh evolve --resume outputs/evolve/<timestamp>
```

## Important Defaults

- `--focus either`
- `--category any`
- `--tom-target-l1 0.30`
- `--tom-target-l2 0.45`
- `--tom-target-l3 0.25`
- `--tom-ratio-tolerance 0.08`
- `--target-pass-rate 20.0`
- `--max-workers 50`

## Notes

- Generated tasks are written to `data/emtom/tasks` by default.
- Run metadata (tier metrics, benchmark logs, generation worker logs, summary) is written under `outputs/evolve/<timestamp>`.
- Benchmark calibration is updated during evolution so each tier adapts to current pool difficulty.
- ToM targets are guidance to generation; they are not hard accept/reject constraints.
