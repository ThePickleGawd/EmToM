python -m habitat_llm.examples.planner_demo --config-name baselines/decentralized_react_baseline.yaml \
    instruction="pick a random number from 1 to 10. the agent with higher number should navigate to the bathroom and explore." \
    mode="cli" \
    evaluation.save_video=True \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=openai_chat \
    trajectory.save=True \
    'trajectory.save_options=["rgb"]' \
    habitat.environment.max_episode_steps=10000 \
