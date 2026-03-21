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
  run_mode: string;
  tested_at: string;
  agent_models: Record<string, string>;
  passed: boolean;
  progress: number;
  steps?: number;
  turns?: number;
  trajectory?: ActionEntry[];
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
  calibration_by_mode?: Record<string, CalibrationMeta>;
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

export interface GenerationWorkerTask {
  task_id: string;
  title: string;
  category: string;
  tom_level?: number;
  success: boolean;
  path: string;
  submitted_at: string;
}

export interface GenerationEvent {
  timestamp: string;
  event_type: string;
  command?: string;
  success?: boolean;
  error?: string | null;
  agent_name?: string;
  model?: string;
  reason?: string;
  num_agents?: number;
  keep?: boolean;
  message?: string;
  scene_id?: string;
  episode_id?: string;
  output_path?: string;
  submitted_count?: number;
  next_required_k_level?: number;
  return_code?: number;
  finished?: boolean;
  failed?: boolean;
  fail_reason?: string;
}

export interface AgentTraceEntry {
  index: number;
  kind: "assistant" | "tool_call" | "tool_result";
  content?: string;
  tool?: string;
  command?: string;
  returncode?: number;
  output?: string;
}

export interface GenerationWorker {
  id: string;
  gpu: number;
  slot: number;
  category: string;
  status: "running" | "finished" | "failed" | "stopped";
  workspace_id: string;
  workspace_path: string;
  worker_log_path: string;
  task_gen_agent?: string;
  task_gen_model?: string;
  submitted_count: number;
  target_tasks: number;
  current_task_index?: number;
  current_k_level?: number;
  scene_id?: string;
  episode_id?: string;
  finished: boolean;
  failed: boolean;
  fail_reason?: string;
  submitted_tasks: GenerationWorkerTask[];
  events: GenerationEvent[];
  agent_trace: AgentTraceEntry[];
  agent_stats?: {
    api_calls?: number;
    instance_cost?: number;
  } | null;
  log_excerpt: {
    head: string[];
    tail: string[];
    line_count: number;
  };
}

export interface GenerationSuccessPoint {
  index: number;
  timestamp: string;
  task_id: string;
  title: string;
  category: string;
  success: boolean;
  worker_id: string;
  cumulative_pass_rate: number;
  cumulative_passed: number;
}

export interface GenerationSummary {
  id: string;
  started_at: string;
  log_dir: string;
  total_workers: number;
  requested_tasks: number;
  submitted_tasks: number;
  finished_workers: number;
  failed_workers: number;
  running_workers: number;
  categories: string[];
}

export interface GenerationIndex {
  generations: GenerationSummary[];
}

export interface GenerationDetail extends GenerationSummary {
  launcher_log?: string;
  success_series: GenerationSuccessPoint[];
  workers: GenerationWorker[];
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
