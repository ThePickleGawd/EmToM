/**
 * Vite dev server plugin that dynamically serves benchmark data
 * by scanning outputs/emtom/ and data/emtom/tasks/ at request time.
 * Replaces the static build-data.py step during development.
 */
import type { Plugin } from "vite";
import fs from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const OUTPUTS_DIR = path.join(PROJECT_ROOT, "outputs", "emtom");
const TASKS_DIR = path.join(PROJECT_ROOT, "data", "emtom", "tasks");
const RESULTS_DIR = path.join(PROJECT_ROOT, "data", "emtom", "results");

function makeRelativePath(absPath: string): string {
  const prefix = PROJECT_ROOT + "/";
  if (absPath.startsWith(prefix)) {
    return "/" + absPath.slice(prefix.length);
  }
  return absPath;
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
  };
}

function flattenCalibration(calibration: any[]): any[] {
  const entries: any[] = [];
  for (const cal of calibration) {
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
  }
  return entries;
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
    instruction[agentId] = (secrets as string[]).join("\n");
  }

  const mechanics = [
    ...new Set(
      (data.mechanic_bindings || [])
        .map((b: any) => b.mechanic_type)
        .filter(Boolean),
    ),
  ];

  const calibration = data.calibration || [];
  const calHistory = calibration.length
    ? flattenCalibration(calibration)
    : [];
  const goldenHistory = flattenGolden(data.golden_trajectory || []);

  const lastCal = calibration.length ? calibration[calibration.length - 1] : null;
  const calPassed = lastCal?.results?.passed || false;
  const calSteps = lastCal?.steps || 0;

  return {
    task_id: taskId,
    task_title: data.title || "",
    task_description: data.task || "",
    category: data.category || "",
    instruction,
    mechanics_active: mechanics,
    steps: calSteps,
    turns: lastCal?.trajectory?.length || 0,
    done: true,
    success: calPassed,
    llm_agents: agents,
    human_agents: [],
    action_history: calHistory,
    golden_trajectory: goldenHistory,
    problem_pddl: data.problem_pddl || "",
    tom_level: data.tom_level,
    tom_reasoning: data.tom_reasoning,
    calibration_meta: lastCal
      ? {
          tested_at: lastCal.tested_at || "",
          agent_models: lastCal.agent_models || {},
          passed: calPassed,
          progress: lastCal.results?.progress || 0,
        }
      : null,
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

          // GET /data/campaign-run/{runKey}.json — campaign run benchmark summary
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
              const summary = JSON.parse(
                fs.readFileSync(summaryFile, "utf-8"),
              );
              // Compute category_stats if missing
              if (!summary.category_stats && Array.isArray(summary.results)) {
                const catStats: Record<
                  string,
                  {
                    total: number;
                    passed: number;
                    pass_rate: number;
                    avg_progress: number;
                    avg_steps: number;
                    timed_out: number;
                  }
                > = {};
                for (const r of summary.results) {
                  const cat = r.category || "unknown";
                  if (!catStats[cat]) {
                    catStats[cat] = {
                      total: 0,
                      passed: 0,
                      pass_rate: 0,
                      avg_progress: 0,
                      avg_steps: 0,
                      timed_out: 0,
                    };
                  }
                  const s = catStats[cat];
                  s.total++;
                  if (r.success) s.passed++;
                  s.avg_steps += r.steps || 0;
                  s.avg_progress +=
                    r.evaluation?.percent_complete ?? (r.success ? 1 : 0);
                  if (r.done === false && r.episode_over === true) s.timed_out++;
                }
                for (const s of Object.values(catStats)) {
                  if (s.total > 0) {
                    s.pass_rate = (s.passed / s.total) * 100;
                    s.avg_steps = s.avg_steps / s.total;
                    s.avg_progress = s.avg_progress / s.total;
                  }
                }
                summary.category_stats = catStats;
              }
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(summary));
            } else {
              res.statusCode = 404;
              res.end("Not found");
            }
            return;
          }

          // GET /data/campaign-task/{runKey}/{taskId}.json — campaign run task detail
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
