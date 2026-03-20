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

export interface LiteralTomProbeResult {
  probe_id: string;
  agent_id: string;
  source_pddl: string;
  question: string;
  supported: boolean;
  status: "passed" | "failed" | "unsupported" | "error";
  raw_response?: string;
  reason?: string;
  details?: {
    expected_response: { predicate: string; holds: boolean; args: string[] };
    parsed_response: { predicate: string; holds: boolean | null; args: string[] };
    fact_true: boolean;
    expected_negated: boolean;
  };
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
  literal_tom_probe_results?: LiteralTomProbeResult[];
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
  campaign_id?: string;
  label?: string;
  status?: "active" | "archived";
  archived_at?: string;
  archive_reason?: string;
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

export interface CampaignIndexEntry {
  campaign_id: string;
  label: string;
  status: "active" | "archived";
  created_at?: string;
  updated_at?: string;
  archived_at?: string;
  archive_reason?: string;
  task_total: number;
  models: string[];
  modes: string[];
}

export interface CampaignIndex {
  active_campaign_id: string | null;
  campaigns: CampaignIndexEntry[];
}

export interface LiteralToMStats {
  literal_tom_score?: number | null;
  literal_tom_task_count?: number;
  literal_tom_probe_count?: number;
  literal_tom_supported_probe_count?: number;
  literal_tom_passed_probe_count?: number;
}

export interface LeaderboardSoloEntry {
  model: string;
  mode: string;
  pass_rate?: number;
  total?: number;
  passed?: number;
  overall?: { pass_rate: number; total: number; passed: number } & LiteralToMStats;
  categories: Record<string, { pass_rate: number; total: number; passed: number } & LiteralToMStats>;
}

export interface LeaderboardMatchupDirection {
  team_0: string;
  team_1: string;
  team_0_wins: number;
  team_1_wins: number;
  draws: number;
  total: number;
}

export interface LeaderboardMatchupCombined {
  model_a_wins: number;
  model_b_wins: number;
  draws: number;
  total: number;
  model_a_win_rate: number;
}

export interface LeaderboardMatchup {
  model_a: string;
  model_b: string;
  mode: string;
  forward?: LeaderboardMatchupDirection;
  swap?: LeaderboardMatchupDirection;
  combined?: LeaderboardMatchupCombined;
}

export interface Leaderboard {
  generated_at: string;
  campaign_id?: string;
  label?: string;
  status?: "active" | "archived";
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
    literal_tom_probe_score?: number | null;
    literal_tom_probe_summary?: {
      probe_count?: number;
      supported_probe_count?: number;
      passed_count?: number;
    };
  };
}

export interface CampaignBenchmarkSummary {
  model: string;
  total: number;
  passed: number;
  failed: number;
  skipped?: number;
  pass_rate: number;
  category_stats: Record<string, {
    total: number;
    passed: number;
    pass_rate: number;
    avg_progress: number;
    avg_steps: number;
    timed_out: number;
  } & LiteralToMStats>;
  literal_tom_score?: number | null;
  literal_tom_task_count?: number;
  literal_tom_probe_count?: number;
  literal_tom_supported_probe_count?: number;
  literal_tom_passed_probe_count?: number;
  results: CampaignRunResult[];
}
