# Deprecated

Evolution is no longer a separate pipeline.

Use the normal generator instead:

```bash
./emtom/run_emtom.sh generate --target-model gpt-5.2 --seed-tasks-dir data/emtom/tasks
```

The `evolve` command remains only as a deprecated alias for `generate`.
