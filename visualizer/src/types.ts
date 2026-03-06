export interface ActionEntry {
  turn: number;
  sim_step: number;
  agent: string;
  action: string;
  result: string;
  skill_steps: number;
  selected_frames: string[];
  frame_paths: string[];
}

export interface TaskDetail {
  task_id: string;
  task_title: string;
  instruction: Record<string, string>;
  mechanics_active: string[];
  steps: number;
  turns: number;
  done: boolean;
  success: boolean;
  llm_agents: string[];
  human_agents: string[];
  action_history: ActionEntry[];
}

export interface TaskSummary {
  task_id: string;
  title: string;
  category: string;
  success: boolean;
  turns: number;
  steps: number;
  agents: number;
}

export interface RunSummary {
  id: string;
  model: string;
  observation_mode: string;
  total: number;
  passed: number;
  pass_rate: number;
  tasks: TaskSummary[];
}

export interface RunsIndex {
  runs: RunSummary[];
}
