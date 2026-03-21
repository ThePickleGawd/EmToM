/**
 * Vite dev server plugin that dynamically serves benchmark and generation data
 * by scanning repo outputs at request time.
 * Replaces the static build-data.py step during development.
 */
import type { Plugin } from "vite";
import fs from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const OUTPUTS_DIR = path.join(PROJECT_ROOT, "outputs", "emtom");
const GENERATIONS_DIR = path.join(PROJECT_ROOT, "outputs", "generations");
const TASKS_DIR = path.join(PROJECT_ROOT, "data", "emtom", "tasks");
const RESULTS_DIR = path.join(PROJECT_ROOT, "data", "emtom", "results");
const ARCHIVES_DIR = path.join(RESULTS_DIR, "archives");

function normalizeCampaignId(rawCampaignId: string): string | null {
  if (!rawCampaignId || rawCampaignId.includes("/") || rawCampaignId.includes("..")) {
    return null;
  }
  return rawCampaignId;
}

function resolveCampaignRoot(rawCampaignId: string): string | null {
  const campaignId = normalizeCampaignId(rawCampaignId);
  if (!campaignId) return null;
  if (campaignId === "active") return RESULTS_DIR;
  return path.join(ARCHIVES_DIR, campaignId);
}

function readJsonIfExists(filePath: string): Record<string, any> | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

function normalizeSecretsText(secrets: unknown): string {
  if (Array.isArray(secrets)) {
    return secrets
      .filter((entry): entry is string => typeof entry === "string")
      .join("\n");
  }
  if (typeof secrets === "string") return secrets;
  return "";
}

function literalTomStats(results: Record<string, any>[]): Record<string, any> {
  let scoredTaskCount = 0;
  let fallbackScoreSum = 0;
  let probeCount = 0;
  let supportedProbeCount = 0;
  let passedProbeCount = 0;

  for (const result of results) {
    if (result.skipped) continue;
    const evaluation = result.evaluation;
    if (!evaluation || typeof evaluation !== "object") continue;

    const probeSummary = evaluation.literal_tom_probe_summary;
    if (probeSummary && typeof probeSummary === "object") {
      const taskProbeCount = Number(probeSummary.probe_count || 0);
      const supported = Number(probeSummary.supported_probe_count || 0);
      const passed = Number(probeSummary.passed_count || 0);
      probeCount += Number.isFinite(taskProbeCount) ? taskProbeCount : 0;
      supportedProbeCount += Number.isFinite(supported) ? supported : 0;
      passedProbeCount += Number.isFinite(passed) ? passed : 0;
      if (supported > 0) {
        scoredTaskCount += 1;
        continue;
      }
    }

    const score = evaluation.literal_tom_probe_score;
    if (typeof score === "number" && Number.isFinite(score)) {
      scoredTaskCount += 1;
      fallbackScoreSum += score;
    }
  }

  let literalTomScore: number | null = null;
  if (supportedProbeCount > 0) {
    literalTomScore = (passedProbeCount / supportedProbeCount) * 100;
  } else if (scoredTaskCount > 0) {
    literalTomScore = (fallbackScoreSum / scoredTaskCount) * 100;
  }

  return {
    literal_tom_score:
      literalTomScore === null ? null : Math.round(literalTomScore * 10) / 10,
    literal_tom_task_count: scoredTaskCount,
    literal_tom_probe_count: probeCount,
    literal_tom_supported_probe_count: supportedProbeCount,
    literal_tom_passed_probe_count: passedProbeCount,
  };
}

function buildCategoryStats(results: Record<string, any>[]): Record<string, any> {
  const grouped: Record<string, Record<string, any>[]> = {};
  for (const result of results) {
    if (result.skipped) continue;
    const category = result.category || "unknown";
    grouped[category] ||= [];
    grouped[category].push(result);
  }

  const categoryStats: Record<string, any> = {};
  for (const [category, categoryResults] of Object.entries(grouped)) {
    const total = categoryResults.length;
    const passed = categoryResults.filter((result) => result.success).length;
    const avgSteps = total
      ? categoryResults.reduce((sum, result) => sum + Number(result.steps || 0), 0) / total
      : 0;
    const avgProgress = total
      ? categoryResults.reduce(
          (sum, result) =>
            sum + Number(result.evaluation?.percent_complete ?? (result.success ? 1 : 0)),
          0,
        ) / total
      : 0;
    const timedOut = categoryResults.filter(
      (result) => result.done === false && result.episode_over === true,
    ).length;

    categoryStats[category] = {
      total,
      passed,
      pass_rate: total ? (passed / total) * 100 : 0,
      avg_progress: avgProgress,
      avg_steps: avgSteps,
      timed_out: timedOut,
      ...literalTomStats(categoryResults),
    };
  }

  return categoryStats;
}

function normalizeSummary(summary: Record<string, any>): Record<string, any> {
  if (!Array.isArray(summary.results)) return summary;

  const results = summary.results;
  const total = results.length;
  const skipped = results.filter((result: Record<string, any>) => result.skipped).length;
  const passed = results.filter((result: Record<string, any>) => result.success).length;
  const evaluated = total - skipped;
  const failed = results.filter(
    (result: Record<string, any>) => !result.skipped && !result.success,
  ).length;

  summary.total = total;
  summary.skipped = skipped;
  summary.passed = passed;
  summary.failed = failed;
  summary.pass_rate = evaluated > 0 ? (passed / evaluated) * 100 : 0;
  summary.category_stats = buildCategoryStats(results);
  Object.assign(summary, literalTomStats(results));
  return summary;
}

function buildCampaignIndex(): Record<string, any> {
  const campaigns: Record<string, any>[] = [];

  const activeCampaign = readJsonIfExists(path.join(RESULTS_DIR, "campaign.json"));
  if (activeCampaign) {
    campaigns.push({
      campaign_id: activeCampaign.campaign_id || "active",
      label: activeCampaign.label || "Active Campaign",
      status: "active",
      created_at: activeCampaign.created_at,
      updated_at: activeCampaign.updated_at,
      archived_at: activeCampaign.archived_at,
      archive_reason: activeCampaign.archive_reason || "",
      task_total: activeCampaign.task_total || 0,
      models: activeCampaign.models || [],
      modes: activeCampaign.modes || [],
    });
  }

  if (fs.existsSync(ARCHIVES_DIR)) {
    const archiveDirs = fs
      .readdirSync(ARCHIVES_DIR)
      .filter((entry) => fs.statSync(path.join(ARCHIVES_DIR, entry)).isDirectory())
      .sort()
      .reverse();
    for (const archiveId of archiveDirs) {
      const campaign = readJsonIfExists(path.join(ARCHIVES_DIR, archiveId, "campaign.json"));
      if (!campaign) continue;
      campaigns.push({
        campaign_id: campaign.campaign_id || archiveId,
        label: campaign.label || archiveId,
        status: "archived",
        created_at: campaign.created_at,
        updated_at: campaign.updated_at,
        archived_at: campaign.archived_at,
        archive_reason: campaign.archive_reason || "",
        task_total: campaign.task_total || 0,
        models: campaign.models || [],
        modes: campaign.modes || [],
      });
    }
  }

  return {
    active_campaign_id: activeCampaign ? (activeCampaign.campaign_id || "active") : null,
    campaigns,
  };
}

function makeRelativePath(absPath: string): string {
  const prefix = PROJECT_ROOT + "/";
  if (absPath.startsWith(prefix)) {
    return "/" + absPath.slice(prefix.length);
  }
  return absPath;
}

function resolveRepoPath(rawPath: string): string {
  if (!rawPath) return "";
  return path.isAbsolute(rawPath) ? rawPath : path.join(PROJECT_ROOT, rawPath);
}

function readJsonLinesIfExists(filePath: string): Record<string, any>[] {
  if (!fs.existsSync(filePath)) return [];
  const lines = fs.readFileSync(filePath, "utf-8").split(/\r?\n/);
  const rows: Record<string, any>[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      rows.push(JSON.parse(trimmed));
    } catch {
      continue;
    }
  }
  return rows;
}

function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*m/g, "");
}

function summarizeLogExcerpt(text: string, headLines = 40): Record<string, any> {
  const lines = stripAnsi(text).split(/\r?\n/).filter((line, index, all) => !(line === "" && index === all.length - 1));
  return {
    head: lines.slice(0, headLines),
    tail: lines,
    line_count: lines.length,
  };
}

function parseGenerationTimestamp(runId: string): string {
  const raw = runId.slice(0, 19);
  const normalized = raw.replace("_", "T").replace(/-/g, (value, offset) => (offset === 13 || offset === 16 ? ":" : value));
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? "" : parsed.toISOString();
}

function normalizeGenerationId(rawRunId: string): string | null {
  if (!rawRunId || rawRunId.includes("/") || rawRunId.includes("..")) {
    return null;
  }
  return rawRunId;
}

function calibrationResult(taskData: Record<string, any>): boolean {
  const calibration = Array.isArray(taskData.calibration) ? taskData.calibration : [];
  let standard =
    calibration.find((entry) => String(entry?.run_mode || "standard") === "standard") ||
    calibration[0];
  if (!standard || typeof standard !== "object") return false;
  const results = standard.results || {};
  if (results.main_goal) return !!results.main_goal.passed;
  if ("winner" in results) return results.winner != null;
  return !!results.passed;
}

function loadSubmittedTaskSummary(taskPathValue: string): Record<string, any> | null {
  const taskPath = resolveRepoPath(taskPathValue);
  const taskData = readJsonIfExists(taskPath);
  if (!taskData) return null;
  let submittedAt = "";
  try {
    submittedAt = fs.statSync(taskPath).mtime.toISOString();
  } catch {
    submittedAt = "";
  }
  return {
    task_id: taskData.task_id || path.basename(taskPath, ".json"),
    title: taskData.title || path.basename(taskPath, ".json"),
    category: taskData.category || "",
    tom_level: taskData.tom_level,
    success: calibrationResult(taskData),
    path: makeRelativePath(taskPath),
    submitted_at: submittedAt,
  };
}

function summarizeGenerationEvent(event: Record<string, any>): Record<string, any> {
  const data =
    event.data && typeof event.data === "object" && !Array.isArray(event.data) ? event.data : {};
  const outputPath =
    typeof data.output_path === "string" && data.output_path
      ? makeRelativePath(resolveRepoPath(data.output_path))
      : undefined;
  const summary: Record<string, any> = {
    timestamp: event.timestamp || "",
    event_type: event.event_type || "",
    command: event.command,
    success: event.success,
    error: event.error,
    agent_name: event.agent_name,
    model: event.model,
    reason: event.reason,
    num_agents: event.num_agents,
    keep: event.keep,
    message: data.message,
    scene_id: data.scene_id || event.scene_id,
    episode_id: data.episode_id || event.episode_id,
    output_path: outputPath,
    submitted_count: data.submitted_count || event.submitted_count,
    next_required_k_level: data.next_required_k_level,
    return_code: event.return_code,
    finished: event.finished,
    failed: event.failed,
    fail_reason: event.fail_reason,
  };
  return Object.fromEntries(
    Object.entries(summary).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  );
}

function parseMiniTrace(tracePathValue: string): Record<string, any> {
  const tracePath = resolveRepoPath(tracePathValue);
  const data = readJsonIfExists(tracePath);
  if (!data) return { entries: [], stats: null };

  const entries: Record<string, any>[] = [];
  const messages = Array.isArray(data.messages) ? data.messages : [];
  for (const [index, message] of messages.entries()) {
    const entryIndex = index + 1;
    if (message.role === "assistant") {
      if (typeof message.content === "string" && message.content.trim()) {
        entries.push({
          index: entryIndex,
          kind: "assistant",
          content: message.content.trim(),
        });
      }
      for (const toolCall of message.tool_calls || []) {
        const rawArguments = toolCall?.function?.arguments;
        let command = "";
        if (typeof rawArguments === "string") {
          try {
            const parsed = JSON.parse(rawArguments);
            command = parsed.command || rawArguments;
          } catch {
            command = rawArguments;
          }
        }
        entries.push({
          index: entryIndex,
          kind: "tool_call",
          tool: toolCall?.function?.name || "",
          command,
        });
      }
      continue;
    }
    if (message.role === "tool" && message.content && typeof message.content === "object") {
      const output =
        message.content.output ??
        [message.content.output_head, message.content.output_tail].filter(Boolean).join("\n");
      entries.push({
        index: entryIndex,
        kind: "tool_result",
        returncode: message.content.returncode,
        output: String(output || "").slice(0, 4000),
      });
    }
  }

  const stats = data.info?.model_stats || {};
  return {
    entries,
    stats: {
      api_calls: stats.api_calls,
      instance_cost: stats.instance_cost,
    },
  };
}

function parseGenerationWorker(workerDir: string): Record<string, any> {
  const snapshot = readJsonIfExists(path.join(workerDir, "worker.json")) || {};
  const events = readJsonLinesIfExists(path.join(workerDir, "events.jsonl")).map(summarizeGenerationEvent);
  const submittedTaskPaths = Array.isArray(snapshot.submitted_tasks) ? snapshot.submitted_tasks : [];
  const submittedTasks = submittedTaskPaths
    .map((taskPath) => (typeof taskPath === "string" ? loadSubmittedTaskSummary(taskPath) : null))
    .filter((task): task is Record<string, any> => !!task);
  const stdoutLogPath = typeof snapshot.stdout_log_path === "string" ? resolveRepoPath(snapshot.stdout_log_path) : "";
  const logText = stdoutLogPath && fs.existsSync(stdoutLogPath) ? fs.readFileSync(stdoutLogPath, "utf-8") : "";
  const tracePath =
    typeof snapshot.agent_trace_path === "string"
      ? snapshot.agent_trace_path
      : path.join(workerDir, "agent_trace.json");
  const agentTrace = parseMiniTrace(tracePath);
  const workspacePath = typeof snapshot.workspace_path === "string" ? resolveRepoPath(snapshot.workspace_path) : "";
  const workerStatus =
    snapshot.status ||
    (snapshot.failed ? "failed" : snapshot.finished ? "finished" : "running");

  return {
    id: snapshot.worker_id || path.basename(workerDir),
    gpu: Number.isFinite(Number(snapshot.gpu)) ? Number(snapshot.gpu) : -1,
    slot: Number.isFinite(Number(snapshot.slot)) ? Number(snapshot.slot) : -1,
    category: snapshot.category || "random",
    status: workerStatus,
    workspace_id: snapshot.workspace_id || (workspacePath ? path.basename(workspacePath) : ""),
    workspace_path: workspacePath ? makeRelativePath(workspacePath) : "",
    worker_log_path: stdoutLogPath ? makeRelativePath(stdoutLogPath) : "",
    task_gen_agent: snapshot.task_gen_agent,
    task_gen_model: snapshot.task_gen_model,
    submitted_count: submittedTasks.length,
    target_tasks: Number(snapshot.target_tasks || snapshot.num_tasks_target || 0),
    current_task_index: snapshot.current_task_index,
    current_k_level: snapshot.current_k_level,
    scene_id: snapshot.scene_id,
    episode_id: snapshot.episode_id,
    finished: !!snapshot.finished,
    failed: !!snapshot.failed,
    fail_reason: snapshot.fail_reason || "",
    submitted_tasks: submittedTasks,
    events,
    agent_trace: agentTrace.entries,
    agent_stats: agentTrace.stats,
    log_excerpt: summarizeLogExcerpt(logText),
  };
}

function buildGenerationSuccessSeries(workers: Record<string, any>[]): Record<string, any>[] {
  const timeline: Record<string, any>[] = [];
  for (const worker of workers) {
    for (const task of worker.submitted_tasks || []) {
      timeline.push({
        timestamp: task.submitted_at || "",
        task_id: task.task_id || "",
        title: task.title || "",
        category: task.category || "",
        success: !!task.success,
        worker_id: worker.id || "",
      });
    }
  }
  timeline.sort((a, b) =>
    `${a.timestamp}|${a.task_id}|${a.worker_id}`.localeCompare(
      `${b.timestamp}|${b.task_id}|${b.worker_id}`,
    ),
  );
  let passed = 0;
  return timeline.map((item, index) => {
    if (item.success) passed += 1;
    return {
      ...item,
      index: index + 1,
      cumulative_pass_rate: passed / (index + 1),
      cumulative_passed: passed,
    };
  });
}

function buildGenerationDetail(runId: string): Record<string, any> | null {
  const normalizedRunId = normalizeGenerationId(runId);
  if (!normalizedRunId) return null;
  const runDir = path.join(GENERATIONS_DIR, normalizedRunId);
  if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) return null;

  const manifest = readJsonIfExists(path.join(runDir, "manifest.json")) || {};
  const workersDir = path.join(runDir, "workers");
  const workers = fs.existsSync(workersDir)
    ? fs
        .readdirSync(workersDir)
        .map((entry) => path.join(workersDir, entry))
        .filter((entry) => fs.statSync(entry).isDirectory())
        .sort()
        .map(parseGenerationWorker)
    : [];
  const successSeries = buildGenerationSuccessSeries(workers);
  const requestedTasks =
    Number(manifest.requested_tasks || 0) ||
    workers.reduce((sum, worker) => sum + Number(worker.target_tasks || 0), 0);
  const submittedTasks = workers.reduce(
    (sum, worker) => sum + (worker.submitted_tasks?.length || 0),
    0,
  );
  const finishedWorkers = workers.filter((worker) => worker.status === "finished").length;
  const failedWorkers = workers.filter((worker) => worker.status === "failed").length;
  const runningWorkers = workers.filter((worker) => worker.status === "running").length;
  const categories = Array.from(
    new Set(workers.map((worker) => worker.category).filter(Boolean)),
  ).sort();
  const launcherLogPath = path.join(runDir, "launcher.log");

  return {
    id: normalizedRunId,
    started_at: manifest.started_at || parseGenerationTimestamp(normalizedRunId),
    log_dir: fs.existsSync(path.join(runDir, "logs")) ? makeRelativePath(path.join(runDir, "logs")) : "",
    launcher_log: fs.existsSync(launcherLogPath) ? makeRelativePath(launcherLogPath) : "",
    total_workers: Number(manifest.total_workers || workers.length),
    requested_tasks: requestedTasks,
    submitted_tasks: submittedTasks,
    finished_workers: finishedWorkers,
    failed_workers: failedWorkers,
    running_workers: runningWorkers,
    categories,
    success_series: successSeries,
    workers,
  };
}

function buildGenerationIndex(): Record<string, any> {
  if (!fs.existsSync(GENERATIONS_DIR)) {
    return { generations: [] };
  }

  const generations = fs
    .readdirSync(GENERATIONS_DIR)
    .map((entry) => path.join(GENERATIONS_DIR, entry))
    .filter((entry) => fs.statSync(entry).isDirectory())
    .sort()
    .reverse()
    .map((entry) => buildGenerationDetail(path.basename(entry)))
    .filter((detail): detail is Record<string, any> => !!detail)
    .map((detail) => ({
      id: detail.id,
      started_at: detail.started_at,
      log_dir: detail.log_dir,
      total_workers: detail.total_workers,
      requested_tasks: detail.requested_tasks,
      submitted_tasks: detail.submitted_tasks,
      finished_workers: detail.finished_workers,
      failed_workers: detail.failed_workers,
      running_workers: detail.running_workers,
      categories: detail.categories,
    }));

  return { generations };
}

function processAction(entry: Record<string, any>): Record<string, any> {
  const framePaths = (entry.selected_frame_paths || []).map(makeRelativePath);
  const out: Record<string, any> = {
    turn: entry.turn,
    sim_step: entry.sim_step,
    agent: entry.agent,
    action: entry.action || "",
    result: entry.result || "",
    skill_steps: entry.skill_steps || 0,
    selected_frames: entry.selected_frames || [],
    frame_paths: framePaths,
  };
  if (entry.thought) out.thought = entry.thought;
  return out;
}

function processTaskDir(taskDir: string): Record<string, any> | null {
  let logFile: string | null = null;

  // Campaign layout: flat planner-log.json in the task dir
  const flatLog = path.join(taskDir, "planner-log.json");
  if (fs.existsSync(flatLog)) {
    logFile = flatLog;
  } else {
    // Benchmark layout: planner-log/planner-log-*.json subdirectory
    const logDir = path.join(taskDir, "planner-log");
    if (!fs.existsSync(logDir)) return null;

    const logFiles = fs
      .readdirSync(logDir)
      .filter((f) => f.startsWith("planner-log-") && f.endsWith(".json"))
      .sort();
    if (logFiles.length === 0) return null;
    logFile = path.join(logDir, logFiles[logFiles.length - 1]);
  }

  if (!logFile) return null;
  let data: Record<string, any>;
  try {
    data = JSON.parse(fs.readFileSync(logFile, "utf-8"));
  } catch {
    return null;
  }

  const actions = (data.action_history || []).map(processAction);

  const evaluation = data.evaluation || {};

  return {
    task_id: data.task_id || path.basename(taskDir),
    task_title: data.task_title || "",
    instruction: data.instruction || {},
    mechanics_active: data.mechanics_active || [],
    steps: data.steps || 0,
    turns: data.turns || 0,
    done: data.done || false,
    success: data.success || false,
    llm_agents: data.llm_agents || [],
    human_agents: data.human_agents || [],
    action_history: actions,
    literal_tom_probe_results: evaluation.literal_tom_probe_results || [],
  };
}

function flattenCalibrationEntry(cal: Record<string, any>): any[] {
  const entries: any[] = [];
  for (const turnData of cal.trajectory || []) {
    const turn = turnData.turn || 0;
    for (const [agentId, agentData] of Object.entries(
      turnData.agents || {},
    )) {
      const ad = agentData as Record<string, any>;
      const entry: Record<string, any> = {
        turn,
        sim_step: turn,
        agent: agentId,
        action: ad.action || "",
        result: ad.observation || "",
        skill_steps: 0,
        selected_frames: [],
        frame_paths: [],
      };
      if (ad.thought) entry.thought = ad.thought;
      entries.push(entry);
    }
  }
  return entries;
}

function calibrationProgress(cal: Record<string, any>): number {
  const results = cal.results || {};
  if (results.main_goal) {
    return results.main_goal.progress || 0;
  }
  if (results.teams) {
    return Math.max(
      0,
      ...Object.values(results.teams).map(
        (team) => ((team as Record<string, any>).progress as number) || 0,
      ),
    );
  }
  return results.progress || 0;
}

function calibrationPassed(cal: Record<string, any>): boolean {
  const results = cal.results || {};
  if (results.main_goal) {
    return !!results.main_goal.passed;
  }
  if ("winner" in results) {
    return !!results.winner;
  }
  return !!results.passed;
}

function flattenGolden(golden: any[]): any[] {
  const entries: any[] = [];
  for (let i = 0; i < golden.length; i++) {
    for (const agentAction of golden[i].actions || []) {
      entries.push({
        turn: i + 1,
        sim_step: i + 1,
        agent: agentAction.agent || "",
        action: agentAction.action || "",
        result: "",
        skill_steps: 0,
        selected_frames: [],
        frame_paths: [],
      });
    }
  }
  return entries;
}

function processTaskFile(taskFile: string): Record<string, any> | null {
  let data: Record<string, any>;
  try {
    data = JSON.parse(fs.readFileSync(taskFile, "utf-8"));
  } catch {
    return null;
  }

  const taskId = data.task_id || path.basename(taskFile, ".json");
  const agents = Array.from(
    { length: data.num_agents || 2 },
    (_, i) => `agent_${i}`,
  );

  const instruction: Record<string, string> = {};
  for (const [agentId, secrets] of Object.entries(
    data.agent_secrets || {},
  )) {
    instruction[agentId] = normalizeSecretsText(secrets);
  }

  const mechanics = [
    ...new Set(
      (data.mechanic_bindings || [])
        .map((b: any) => b.mechanic_type)
        .filter(Boolean),
    ),
  ];

  const calibration = data.calibration || [];
  const goldenHistory = flattenGolden(data.golden_trajectory || []);
  const calibrationByMode: Record<string, any> = {};
  for (const cal of calibration) {
    const runMode = String(cal.run_mode || "standard");
    calibrationByMode[runMode] = {
      run_mode: runMode,
      tested_at: cal.tested_at || "",
      agent_models: cal.agent_models || {},
      passed: calibrationPassed(cal),
      progress: calibrationProgress(cal),
      steps: cal.steps || 0,
      turns: (cal.trajectory || []).length,
      trajectory: flattenCalibrationEntry(cal),
    };
  }

  const defaultMode = calibrationByMode.standard
    ? "standard"
    : Object.keys(calibrationByMode)[0] || null;
  const defaultCal = defaultMode ? calibrationByMode[defaultMode] : null;
  const calHistory = defaultCal?.trajectory || [];
  const calPassed = defaultCal?.passed || false;
  const calSteps = defaultCal?.steps || 0;

  return {
    task_id: taskId,
    task_title: data.title || "",
    task_description: data.task || "",
    category: data.category || "",
    instruction,
    mechanics_active: mechanics,
    steps: calSteps,
    turns: defaultCal?.turns || 0,
    done: true,
    success: calPassed,
    llm_agents: agents,
    human_agents: [],
    action_history: calHistory,
    golden_trajectory: goldenHistory,
    problem_pddl: data.problem_pddl || "",
    tom_level: data.tom_level,
    tom_reasoning: data.tom_reasoning,
    calibration_meta: defaultCal,
    calibration_by_mode: calibrationByMode,
  };
}

/**
 * Find the "results" directory for a benchmark run, handling two layouts:
 *  - Flat: {runDir}/results/{taskId}/planner-log/...
 *  - Nested: {runDir}/{wrapperDir}/benchmark-{N}agents/results/{taskId}/planner-log/...
 *
 * Returns an array of { resultsDir, summary } for each results dir found.
 */
function findResultsDirs(
  runDir: string,
): { resultsDir: string; summary: Record<string, any> }[] {
  // Flat layout: {runDir}/results/
  const flatResults = path.join(runDir, "results");
  if (fs.existsSync(flatResults) && fs.statSync(flatResults).isDirectory()) {
    let summary: Record<string, any> = {};
    const summaryFile = path.join(flatResults, "benchmark_summary.json");
    if (fs.existsSync(summaryFile)) {
      try {
        summary = JSON.parse(fs.readFileSync(summaryFile, "utf-8"));
      } catch {}
    }
    return [{ resultsDir: flatResults, summary }];
  }

  // Nested layout: {runDir}/{wrapper}/benchmark-*/results/
  const out: { resultsDir: string; summary: Record<string, any> }[] = [];
  let entries: string[];
  try {
    entries = fs.readdirSync(runDir);
  } catch {
    return out;
  }
  for (const wrapper of entries) {
    const wrapperDir = path.join(runDir, wrapper);
    if (!fs.statSync(wrapperDir).isDirectory() || wrapper.startsWith(".") || wrapper === "logs")
      continue;
    let benchDirs: string[];
    try {
      benchDirs = fs
        .readdirSync(wrapperDir)
        .filter((d) => d.startsWith("benchmark-"));
    } catch {
      continue;
    }
    for (const bd of benchDirs) {
      const resultsDir = path.join(wrapperDir, bd, "results");
      if (!fs.existsSync(resultsDir) || !fs.statSync(resultsDir).isDirectory())
        continue;
      let summary: Record<string, any> = {};
      const summaryFile = path.join(resultsDir, "benchmark_summary.json");
      if (fs.existsSync(summaryFile)) {
        try {
          summary = JSON.parse(fs.readFileSync(summaryFile, "utf-8"));
        } catch {}
      }
      out.push({ resultsDir, summary });
    }
  }
  return out;
}

function processRun(runDir: string): Record<string, any> | null {
  const resultsDirs = findResultsDirs(runDir);
  if (resultsDirs.length === 0) return null;

  const runId = path.basename(runDir);
  const taskSummaries: any[] = [];
  let mergedSummary: Record<string, any> = {};

  for (const { resultsDir, summary } of resultsDirs) {
    // Use the first non-empty summary for top-level model info
    if (!mergedSummary.model && summary.model) {
      mergedSummary = summary;
    }

    const entries = fs.readdirSync(resultsDir).sort();
    for (const entry of entries) {
      const taskDir = path.join(resultsDir, entry);
      if (
        !fs.statSync(taskDir).isDirectory() ||
        entry.startsWith(".") ||
        entry === "benchmark_summary.json"
      )
        continue;

      const taskData = processTaskDir(taskDir);
      if (!taskData) continue;

      let category = "";
      if (summary.results) {
        const match = (summary.results as any[]).find(
          (r) => r.task_id === taskData.task_id,
        );
        if (match) category = match.category || "";
      }

      taskSummaries.push({
        task_id: taskData.task_id,
        title: taskData.task_title,
        category,
        success: taskData.success,
        turns: taskData.turns,
        steps: taskData.steps,
        agents: taskData.llm_agents.length,
      });
    }
  }

  if (taskSummaries.length === 0) return null;

  return {
    id: runId,
    model: mergedSummary.model || "",
    observation_mode: mergedSummary.benchmark_observation_mode || "",
    total: taskSummaries.length,
    passed: taskSummaries.filter((t) => t.success).length,
    pass_rate:
      taskSummaries.length > 0
        ? (taskSummaries.filter((t) => t.success).length / taskSummaries.length) * 100
        : 0,
    tasks: taskSummaries,
  };
}

function buildRunsIndex(): Record<string, any> {
  const runs: any[] = [];

  if (fs.existsSync(OUTPUTS_DIR)) {
    const runDirs = fs
      .readdirSync(OUTPUTS_DIR)
      .filter((d) => {
        const full = path.join(OUTPUTS_DIR, d);
        return fs.statSync(full).isDirectory() && d.includes("benchmark");
      })
      .sort()
      .reverse();

    for (const dir of runDirs) {
      const runData = processRun(path.join(OUTPUTS_DIR, dir));
      if (runData) runs.push(runData);
    }
  }

  const library: any[] = [];
  if (fs.existsSync(TASKS_DIR)) {
    const taskFiles = fs
      .readdirSync(TASKS_DIR)
      .filter((f) => f.endsWith(".json"))
      .sort()
      .reverse();

    for (const tf of taskFiles) {
      const taskData = processTaskFile(path.join(TASKS_DIR, tf));
      if (!taskData) continue;
      library.push({
        task_id: taskData.task_id,
        title: taskData.task_title,
        category: taskData.category,
        success: taskData.success,
        turns: taskData.turns,
        steps: taskData.steps,
        agents: taskData.llm_agents.length,
      });
    }
  }

  return { runs, library };
}

/**
 * Find a library task file by task_id, handling timestamp-prefixed filenames.
 * Scans files in TASKS_DIR and reads each to match the task_id field.
 */
function findLibraryFile(taskId: string): string | null {
  if (!fs.existsSync(TASKS_DIR)) return null;
  // Fast path: check if any filename ends with the task_id
  const files = fs.readdirSync(TASKS_DIR).filter((f) => f.endsWith(".json"));
  for (const f of files) {
    if (f === `${taskId}.json` || f.endsWith(`_${taskId}.json`)) {
      return path.join(TASKS_DIR, f);
    }
  }
  // Slow path: read each file and check task_id field
  for (const f of files) {
    const filePath = path.join(TASKS_DIR, f);
    try {
      const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
      if (data.task_id === taskId) return filePath;
    } catch {
      continue;
    }
  }
  return null;
}

export default function dynamicDataPlugin(): Plugin {
  return {
    name: "dynamic-data",
    configureServer(server) {
      server.middlewares.use("/data", (req, res, next) => {
        try {
          const urlPath = decodeURIComponent(req.url || "");

          // GET /data/runs.json — dynamic index
          if (urlPath === "/runs.json") {
            const index = buildRunsIndex();
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(index));
            return;
          }

          if (urlPath === "/campaign-index.json") {
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(buildCampaignIndex()));
            return;
          }

          if (urlPath === "/generation-index.json") {
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(buildGenerationIndex()));
            return;
          }

          const generationDetailMatch = urlPath.match(
            /^\/generation\/([^/]+)\.json$/,
          );
          if (generationDetailMatch) {
            const detail = buildGenerationDetail(generationDetailMatch[1]);
            if (!detail) {
              res.statusCode = 404;
              res.end("Not found");
              return;
            }
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(detail));
            return;
          }

          // GET /data/tasks/_library/{taskId}.json — library task detail
          const libMatch = urlPath.match(
            /^\/tasks\/_library\/(.+)\.json$/,
          );
          if (libMatch) {
            const taskId = libMatch[1];
            // Try exact filename first, then scan for matching task_id
            // (filenames may have a timestamp prefix like 20260307_1756_emtom-...)
            const exactFile = path.join(TASKS_DIR, `${taskId}.json`);
            const taskFile = fs.existsSync(exactFile)
              ? exactFile
              : findLibraryFile(taskId);
            if (taskFile) {
              const taskData = processTaskFile(taskFile);
              if (taskData) {
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(taskData));
                return;
              }
            }
            res.statusCode = 404;
            res.end("Not found");
            return;
          }

          // GET /data/campaign.json — campaign definition
          if (urlPath === "/campaign.json") {
            const campaignFile = path.join(RESULTS_DIR, "campaign.json");
            if (fs.existsSync(campaignFile)) {
              res.setHeader("Content-Type", "application/json");
              res.end(fs.readFileSync(campaignFile, "utf-8"));
            } else {
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(null));
            }
            return;
          }

          const scopedCampaignMatch = urlPath.match(
            /^\/campaigns\/([^/]+)\/campaign\.json$/,
          );
          if (scopedCampaignMatch) {
            const campaignRoot = resolveCampaignRoot(scopedCampaignMatch[1]);
            if (!campaignRoot) {
              res.statusCode = 400;
              res.end("Invalid campaign id");
              return;
            }
            const campaignFile = path.join(campaignRoot, "campaign.json");
            const campaign = readJsonIfExists(campaignFile);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(campaign));
            return;
          }

          // GET /data/leaderboard.json — leaderboard
          if (urlPath === "/leaderboard.json") {
            const lbFile = path.join(RESULTS_DIR, "leaderboard.json");
            if (fs.existsSync(lbFile)) {
              res.setHeader("Content-Type", "application/json");
              res.end(fs.readFileSync(lbFile, "utf-8"));
            } else {
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(null));
            }
            return;
          }

          const scopedLeaderboardMatch = urlPath.match(
            /^\/campaigns\/([^/]+)\/leaderboard\.json$/,
          );
          if (scopedLeaderboardMatch) {
            const campaignRoot = resolveCampaignRoot(scopedLeaderboardMatch[1]);
            if (!campaignRoot) {
              res.statusCode = 400;
              res.end("Invalid campaign id");
              return;
            }
            const leaderboard = readJsonIfExists(path.join(campaignRoot, "leaderboard.json"));
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(leaderboard));
            return;
          }

          // GET /data/campaign-run/{runKey}.json — campaign run benchmark summary
          const scopedCampaignRunMatch = urlPath.match(
            /^\/campaign-run\/([^/]+)\/(.+)\.json$/,
          );
          if (scopedCampaignRunMatch) {
            const [, campaignId, runKey] = scopedCampaignRunMatch;
            const campaignRoot = resolveCampaignRoot(campaignId);
            if (!campaignRoot) {
              res.statusCode = 400;
              res.end("Invalid campaign id");
              return;
            }
            const summaryFile = path.join(
              campaignRoot,
              "runs",
              runKey,
              "benchmark_summary.json",
            );
            const summary = readJsonIfExists(summaryFile);
            if (summary) {
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(normalizeSummary(summary)));
            } else {
              res.statusCode = 404;
              res.end("Not found");
            }
            return;
          }

          const campaignRunMatch = urlPath.match(
            /^\/campaign-run\/(.+)\.json$/,
          );
          if (campaignRunMatch) {
            const runKey = campaignRunMatch[1];
            const summaryFile = path.join(
              RESULTS_DIR,
              "runs",
              runKey,
              "benchmark_summary.json",
            );
            if (fs.existsSync(summaryFile)) {
              const summary = normalizeSummary(
                JSON.parse(fs.readFileSync(summaryFile, "utf-8")),
              );
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(summary));
            } else {
              res.statusCode = 404;
              res.end("Not found");
            }
            return;
          }

          // GET /data/campaign-task/{runKey}/{taskId}.json — campaign run task detail
          const scopedCampaignTaskMatch = urlPath.match(
            /^\/campaign-task\/([^/]+)\/([^/]+)\/(.+)\.json$/,
          );
          if (scopedCampaignTaskMatch) {
            const [, campaignId, runKey, taskId] = scopedCampaignTaskMatch;
            const campaignRoot = resolveCampaignRoot(campaignId);
            if (!campaignRoot) {
              res.statusCode = 400;
              res.end("Invalid campaign id");
              return;
            }
            const taskDir = path.join(
              campaignRoot,
              "runs",
              runKey,
              "tasks",
              taskId,
            );
            if (fs.existsSync(taskDir)) {
              const taskData = processTaskDir(taskDir);
              if (taskData) {
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(taskData));
                return;
              }
            }
            const campaign = readJsonIfExists(path.join(campaignRoot, "campaign.json"));
            const runDef = campaign?.runs?.[runKey];
            if (runDef?.output_dir) {
              const outputRunDir = path.join(PROJECT_ROOT, runDef.output_dir);
              const resultsDirs = findResultsDirs(outputRunDir);
              for (const { resultsDir } of resultsDirs) {
                const td = path.join(resultsDir, taskId);
                if (fs.existsSync(td)) {
                  const taskData = processTaskDir(td);
                  if (taskData) {
                    res.setHeader("Content-Type", "application/json");
                    res.end(JSON.stringify(taskData));
                    return;
                  }
                }
              }
            }
            res.statusCode = 404;
            res.end("Not found");
            return;
          }

          const campaignTaskMatch = urlPath.match(
            /^\/campaign-task\/([^/]+)\/(.+)\.json$/,
          );
          if (campaignTaskMatch) {
            const [, runKey, taskId] = campaignTaskMatch;
            const taskDir = path.join(
              RESULTS_DIR,
              "runs",
              runKey,
              "tasks",
              taskId,
            );
            if (fs.existsSync(taskDir)) {
              const taskData = processTaskDir(taskDir);
              if (taskData) {
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(taskData));
                return;
              }
            }
            // Fallback: check outputs dir for campaign runs
            const campaignFile = path.join(RESULTS_DIR, "campaign.json");
            if (fs.existsSync(campaignFile)) {
              try {
                const campaign = JSON.parse(
                  fs.readFileSync(campaignFile, "utf-8"),
                );
                const runDef = campaign.runs?.[runKey];
                if (runDef?.output_dir) {
                  const outputRunDir = path.join(
                    PROJECT_ROOT,
                    runDef.output_dir,
                  );
                  const resultsDirs = findResultsDirs(outputRunDir);
                  for (const { resultsDir } of resultsDirs) {
                    const td = path.join(resultsDir, taskId);
                    if (fs.existsSync(td)) {
                      const taskData = processTaskDir(td);
                      if (taskData) {
                        res.setHeader("Content-Type", "application/json");
                        res.end(JSON.stringify(taskData));
                        return;
                      }
                    }
                  }
                }
              } catch {}
            }
            res.statusCode = 404;
            res.end("Not found");
            return;
          }

          // GET /data/tasks/{runId}/{taskId}.json — benchmark task detail
          const taskMatch = urlPath.match(
            /^\/tasks\/([^/]+)\/(.+)\.json$/,
          );
          if (taskMatch) {
            const [, runId, taskId] = taskMatch;
            const runDir = path.join(OUTPUTS_DIR, runId);
            if (fs.existsSync(runDir)) {
              // Search all results dirs (flat + nested) for this task
              const resultsDirs = findResultsDirs(runDir);
              for (const { resultsDir } of resultsDirs) {
                const taskDir = path.join(resultsDir, taskId);
                if (fs.existsSync(taskDir)) {
                  const taskData = processTaskDir(taskDir);
                  if (taskData) {
                    res.setHeader("Content-Type", "application/json");
                    res.end(JSON.stringify(taskData));
                    return;
                  }
                }
              }
            }
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
        } catch (err) {
          console.error("[dynamic-data]", err);
          res.statusCode = 500;
          res.end(JSON.stringify({ error: String(err) }));
          return;
        }

        next();
      });
    },
  };
}
