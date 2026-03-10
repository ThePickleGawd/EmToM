export interface ActionEntry {
  turn: number;
  sim_step: number;
  agent: string;
  action: string;
  result: string;
  skill_steps: number;
  selected_frames: string[];
  frame_paths: string[];
  thought?: string;
}

export interface CalibrationMeta {
  tested_at: string;
  agent_models: Record<string, string>;
  passed: boolean;
  progress: number;
}

export interface TaskDetail {
  task_id: string;
  task_title: string;
  task_description?: string;
  instruction: Record<string, string>;
  mechanics_active: string[];
  steps: number;
  turns: number;
  done: boolean;
  success: boolean;
  llm_agents: string[];
  human_agents: string[];
  action_history: ActionEntry[];
  golden_trajectory?: ActionEntry[];
  problem_pddl?: string;
  tom_level?: number;
  tom_reasoning?: string;
  calibration_meta?: CalibrationMeta | null;
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
  library: TaskSummary[];
}

/* ─── Campaign types ─── */

export interface CampaignRunDef {
  type: "solo" | "matchup";
  status: "pending" | "complete" | "failed";
  mode: string;
  category: string;
  // solo
  model?: string;
  // matchup
  model_a?: string;
  model_b?: string;
  team_0?: string;
  team_1?: string;
  direction?: "forward" | "swap";
  // results (attached after completion)
  output_dir?: string;
}

export interface Campaign {
  created_at: string;
  updated_at: string;
  models: string[];
  modes: string[];
  competitive_matchups: [string, string][];
  tasks_dir: string;
  task_counts: Record<string, number>;
  task_total: number;
  runs: Record<string, CampaignRunDef>;
}

export interface LeaderboardSoloEntry {
  pass_rate: number;
  total: number;
  passed: number;
  categories: Record<string, { pass_rate: number; total: number; passed: number }>;
}

export interface LeaderboardMatchup {
  model_a: string;
  model_b: string;
  model_a_wins: number;
  model_b_wins: number;
  draws: number;
  total: number;
  model_a_win_rate: number;
}

export interface Leaderboard {
  generated_at: string;
  models: string[];
  modes: string[];
  solo: Record<string, LeaderboardSoloEntry>;
  matchups: Record<string, LeaderboardMatchup>;
}

export interface CampaignRunResult {
  task_id: string;
  title: string;
  category: string;
  success: boolean;
  steps: number;
  turns: number;
  evaluation?: {
    success: boolean;
    percent_complete: number;
    completed_required: number;
  };
}

export interface CampaignBenchmarkSummary {
  model: string;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  category_stats: Record<string, {
    total: number;
    passed: number;
    pass_rate: number;
    avg_progress: number;
    avg_steps: number;
    timed_out: number;
  }>;
  results: CampaignRunResult[];
}
