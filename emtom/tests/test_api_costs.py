import json

from emtom.api_costs import summarize_worker_costs


def test_summarize_worker_costs_merges_trace_and_usage_log(tmp_path):
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()

    (worker_dir / "api_usage.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "model": "gpt-5.2",
                        "api_calls": 1,
                        "input_tokens": 1000,
                        "output_tokens": 200,
                        "cached_input_tokens": 100,
                        "cost": 0.01,
                        "source": "judge",
                    }
                ),
                json.dumps(
                    {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "api_calls": 1,
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "cached_input_tokens": 0,
                        "cost": 0.02,
                        "source": "test_task",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    trace_payload = {
        "info": {"model_stats": {"instance_cost": 0.03, "api_calls": 2}},
        "messages": [
            {
                "extra": {
                    "response": {
                        "model": "gpt-5.2-2025-12-11",
                        "usage": {
                            "prompt_tokens": 200,
                            "completion_tokens": 20,
                            "prompt_tokens_details": {"cached_tokens": 10},
                        },
                    },
                    "cost": 0.01,
                }
            },
            {
                "extra": {
                    "response": {
                        "model": "gpt-5.2-2025-12-11",
                        "usage": {
                            "prompt_tokens": 300,
                            "completion_tokens": 30,
                            "prompt_tokens_details": {"cached_tokens": 15},
                        },
                    },
                    "cost": 0.02,
                }
            },
        ],
    }
    (worker_dir / "agent_trace.json").write_text(
        json.dumps(trace_payload),
        encoding="utf-8",
    )

    summary = summarize_worker_costs(worker_dir)

    assert summary["total_api_calls"] == 4
    assert abs(summary["total_cost"] - 0.06) < 1e-9
    assert set(summary["models"].keys()) == {"gpt-5.2", "gpt-5.4"}
    assert summary["models"]["gpt-5.2"]["api_calls"] == 3
    assert summary["models"]["gpt-5.2"]["input_tokens"] == 1500
    assert summary["models"]["gpt-5.2"]["output_tokens"] == 250
    assert summary["models"]["gpt-5.2"]["cached_input_tokens"] == 125
