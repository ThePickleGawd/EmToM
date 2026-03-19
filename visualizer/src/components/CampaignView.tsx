import { useState, useEffect, useCallback } from "react";
import type {
  Campaign,
  CampaignIndex,
  CampaignIndexEntry,
  Leaderboard,
  LeaderboardMatchup,
  CampaignBenchmarkSummary,
  CampaignRunResult,
  TaskDetail,
  LiteralToMStats,
} from "../types";
import TaskView from "./TaskView";
import { downloadJson } from "../download";

interface Props {
  onImageClick: (src: string) => void;
}

type Panel = "leaderboard" | "model-detail" | "run-detail" | "task-detail";

interface ModelDetailSelection {
  model: string;
  mode: string;
}

type SummaryCache = Record<string, CampaignBenchmarkSummary | null | undefined>;
type TaskParentPanel = "model-detail" | "run-detail";

const MODEL_COLORS: Record<string, string> = {
  "gpt-5.2": "#10b981",
  "kimi-k2.5": "#f59e0b",
  "qwen-3.5": "#8b5cf6",
};

function modelColor(model: string): string {
  return MODEL_COLORS[model] || "#6b7280";
}

function statusDot(status: string) {
  const color =
    status === "complete"
      ? "var(--success)"
      : status === "failed"
        ? "var(--failure)"
        : "var(--text-muted)";
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        marginRight: 6,
        flexShrink: 0,
      }}
    />
  );
}

function formatLiteralTomScore(stats?: LiteralToMStats | null): string {
  if (typeof stats?.literal_tom_score !== "number") {
    return "--";
  }
  return `${stats.literal_tom_score.toFixed(0)}%`;
}

function formatLiteralTomDetail(stats?: LiteralToMStats | null): string {
  if (typeof stats?.literal_tom_score !== "number") {
    return "no probes";
  }
  return `${stats.literal_tom_score.toFixed(1)}% · ${stats.literal_tom_passed_probe_count ?? 0}/${stats.literal_tom_supported_probe_count ?? 0} probes`;
}

function literalTomFromEvaluation(
  evaluation?: CampaignRunResult["evaluation"],
): LiteralToMStats | undefined {
  if (!evaluation) return undefined;
  return {
    literal_tom_score:
      typeof evaluation.literal_tom_probe_score === "number"
        ? evaluation.literal_tom_probe_score * 100
        : null,
    literal_tom_probe_count: evaluation.literal_tom_probe_summary?.probe_count,
    literal_tom_supported_probe_count:
      evaluation.literal_tom_probe_summary?.supported_probe_count,
    literal_tom_passed_probe_count:
      evaluation.literal_tom_probe_summary?.passed_count,
  };
}

export default function CampaignView({ onImageClick }: Props) {
  const [campaignIndex, setCampaignIndex] = useState<CampaignIndex | null>(null);
  const [selectedCampaignId, setSelectedCampaignId] = useState("active");
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [leaderboard, setLeaderboard] = useState<Leaderboard | null>(null);
  const [panel, setPanel] = useState<Panel>("leaderboard");
  const [selectedModelDetail, setSelectedModelDetail] =
    useState<ModelDetailSelection | null>(null);
  const [taskParentPanel, setTaskParentPanel] =
    useState<TaskParentPanel>("run-detail");
  const [selectedRunKey, setSelectedRunKey] = useState("");
  const [runSummary, setRunSummary] = useState<CampaignBenchmarkSummary | null>(
    null,
  );
  const [taskDetail, setTaskDetail] = useState<TaskDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [summaryCache, setSummaryCache] = useState<SummaryCache>({});
  const [downloadingRunKey, setDownloadingRunKey] = useState<string | null>(
    null,
  );
  const [downloadingTaskKey, setDownloadingTaskKey] = useState<string | null>(
    null,
  );
  const [downloadingEverything, setDownloadingEverything] = useState(false);

  useEffect(() => {
    fetch("/data/campaign-index.json")
      .then((r) => r.json())
      .then((data: CampaignIndex) => {
        setCampaignIndex(data);
        const initialId = data.active_campaign_id ?? data.campaigns[0]?.campaign_id ?? "";
        if (!initialId) {
          setCampaign(null);
          setLeaderboard(null);
          setLoading(false);
          return;
        }
        setSelectedCampaignId(initialId);
      })
      .catch(() => {
        setCampaignIndex({ active_campaign_id: null, campaigns: [] });
        setCampaign(null);
        setLeaderboard(null);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!selectedCampaignId) return;
    setLoading(true);
    Promise.all([
      fetch(`/data/campaigns/${encodeURIComponent(selectedCampaignId)}/campaign.json`).then((r) => r.json()),
      fetch(`/data/campaigns/${encodeURIComponent(selectedCampaignId)}/leaderboard.json`).then((r) => r.json()),
    ])
      .then(([c, l]) => {
        setCampaign(c);
        setLeaderboard(l);
        setPanel("leaderboard");
        setSelectedModelDetail(null);
        setSelectedRunKey("");
        setRunSummary(null);
        setTaskDetail(null);
        setLoading(false);
      })
      .catch(() => {
        setCampaign(null);
        setLeaderboard(null);
        setLoading(false);
      });
  }, [selectedCampaignId]);

  const fetchCampaignTask = useCallback(
    async (runKey: string, taskId: string): Promise<TaskDetail> => {
      const response = await fetch(
        `/data/campaign-task/${encodeURIComponent(selectedCampaignId)}/${encodeURIComponent(runKey)}/${encodeURIComponent(taskId)}.json`,
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
    [selectedCampaignId],
  );

  const loadRunSummary = useCallback(
    async (runKey: string): Promise<CampaignBenchmarkSummary | null> => {
      const cacheKey = `${selectedCampaignId}:${runKey}`;
      if (summaryCache[cacheKey] !== undefined) {
        return summaryCache[cacheKey] ?? null;
      }
      try {
        const response = await fetch(
          `/data/campaign-run/${encodeURIComponent(selectedCampaignId)}/${encodeURIComponent(runKey)}.json`,
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data: CampaignBenchmarkSummary = await response.json();
        setSummaryCache((prev) => ({ ...prev, [cacheKey]: data }));
        return data;
      } catch (error) {
        console.error("Failed to load run summary", runKey, error);
        setSummaryCache((prev) => ({ ...prev, [cacheKey]: null }));
        return null;
      }
    },
    [selectedCampaignId, summaryCache],
  );

  const openRun = useCallback((runKey: string) => {
    const cacheKey = `${selectedCampaignId}:${runKey}`;
    setSelectedRunKey(runKey);
    setSelectedModelDetail(null);
    setPanel("run-detail");
    setRunSummary(summaryCache[cacheKey] ?? null);
    loadRunSummary(runKey).then(setRunSummary);
  }, [loadRunSummary, selectedCampaignId, summaryCache]);

  const openTask = useCallback(
    (runKey: string, taskId: string, parentPanel: TaskParentPanel) => {
      setSelectedRunKey(runKey);
      setTaskParentPanel(parentPanel);
      setTaskDetail(null);
      setPanel("task-detail");
      fetchCampaignTask(runKey, taskId)
        .then(setTaskDetail)
        .catch(() => setTaskDetail(null));
    },
    [fetchCampaignTask, selectedCampaignId],
  );

  const openModelDetail = useCallback(
    async (model: string, mode: string) => {
      if (!campaign) return;
      setSelectedModelDetail({ model, mode });
      setPanel("model-detail");
      const matchingRunKeys = Object.entries(campaign.runs)
        .filter(
          ([, run]) =>
            run.type === "solo" &&
            run.model === model &&
            run.mode === mode &&
            run.status === "complete",
        )
        .map(([runKey]) => runKey);
      await Promise.all(matchingRunKeys.map((runKey) => loadRunSummary(runKey)));
    },
    [campaign, loadRunSummary],
  );

  const downloadTask = useCallback(
    async (runKey: string, taskId: string) => {
      const downloadKey = `${runKey}:${taskId}`;
      setDownloadingTaskKey(downloadKey);
      try {
        const task = await fetchCampaignTask(runKey, taskId);
        downloadJson(task, `${selectedCampaignId}-${taskId}.json`);
      } catch (error) {
        console.error("Failed to download task trajectory", runKey, taskId, error);
      } finally {
        setDownloadingTaskKey(null);
      }
    },
    [fetchCampaignTask, selectedCampaignId],
  );

  const downloadRunTasks = useCallback(
    async (
      runKey: string,
      runDef: Campaign["runs"][string],
      summary: CampaignBenchmarkSummary,
    ) => {
      setDownloadingRunKey(runKey);
      try {
        const tasks = await Promise.all(
          summary.results.map(async (result) => ({
            task_id: result.task_id,
            task: await fetchCampaignTask(runKey, result.task_id),
          })),
        );
        downloadJson(
          {
            run_key: runKey,
            run: runDef,
            summary,
            tasks,
          },
          `${selectedCampaignId}-${runKey}-trajectories.json`,
        );
      } catch (error) {
        console.error("Failed to download run trajectories", runKey, error);
      } finally {
        setDownloadingRunKey(null);
      }
    },
    [fetchCampaignTask],
  );

  const downloadEverything = useCallback(async () => {
    if (!campaign) return;
    setDownloadingEverything(true);
    try {
      const completedRuns = Object.entries(campaign.runs).filter(
        ([, run]) => run.status === "complete",
      );
      const runs = await Promise.all(
        completedRuns.map(async ([runKey, runDef]) => {
          const summary = await loadRunSummary(runKey);
          const tasks = summary
            ? await Promise.all(
                summary.results.map(async (result) => ({
                  task_id: result.task_id,
                  task: await fetchCampaignTask(runKey, result.task_id),
                })),
              )
            : [];
          return {
            run_key: runKey,
            run: runDef,
            summary,
            tasks,
          };
        }),
      );
      downloadJson(
        {
          campaign,
          leaderboard,
          runs,
        },
        `${selectedCampaignId}-emtom-campaign-everything.json`,
      );
    } catch (error) {
      console.error("Failed to download full campaign bundle", error);
    } finally {
      setDownloadingEverything(false);
    }
  }, [campaign, fetchCampaignTask, leaderboard, loadRunSummary, selectedCampaignId]);

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner" />
        Loading campaign...
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="campaign-empty">
        <div className="campaign-empty-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M3 9h18M9 21V9" />
          </svg>
        </div>
        <p>No campaign configured.</p>
        <code className="campaign-empty-cmd">
          ./emtom/run_emtom.sh campaign add --models gpt-5.2 kimi-k2.5 --modes text vision
        </code>
      </div>
    );
  }

  const selectedCampaignMeta = campaignIndex?.campaigns.find(
    (entry) => entry.campaign_id === selectedCampaignId,
  );
  const runEntries = Object.entries(campaign.runs);
  const soloRuns = runEntries.filter(([, r]) => r.type === "solo");
  const matchupRuns = runEntries.filter(([, r]) => r.type === "matchup");

  return (
    <div className="campaign-root">
      {/* Breadcrumb nav */}
      <div className="campaign-nav">
        <button
          className={`campaign-nav-btn ${panel === "leaderboard" ? "active" : ""}`}
          onClick={() => setPanel("leaderboard")}
        >
          Leaderboard
        </button>
        {panel === "model-detail" && selectedModelDetail && (
          <>
            <span className="campaign-nav-sep">/</span>
            <span className="campaign-nav-current">
              {selectedModelDetail.model} / {selectedModelDetail.mode}
            </span>
          </>
        )}
        {panel === "run-detail" && (
          <>
            <span className="campaign-nav-sep">/</span>
            <span className="campaign-nav-current">{selectedRunKey}</span>
          </>
        )}
        {panel === "task-detail" && (
          <>
            <span className="campaign-nav-sep">/</span>
            <button
              className="campaign-nav-btn"
              onClick={() =>
                setPanel(taskParentPanel)
              }
            >
              {taskParentPanel === "model-detail" && selectedModelDetail
                ? `${selectedModelDetail.model} / ${selectedModelDetail.mode}`
                : selectedRunKey}
            </button>
            <span className="campaign-nav-sep">/</span>
            <span className="campaign-nav-current">Task</span>
          </>
        )}
      </div>

      {/* Campaign header stats */}
      <div className="campaign-header">
        <div className="campaign-header-actions">
          {campaignIndex && campaignIndex.campaigns.length > 0 && (
            <label className="campaign-selector-wrap">
              <span className="campaign-selector-label">Campaign</span>
              <select
                className="campaign-selector"
                value={selectedCampaignId}
                onChange={(event) => setSelectedCampaignId(event.target.value)}
              >
                {campaignIndex.campaigns.map((entry) => (
                  <option key={entry.campaign_id} value={entry.campaign_id}>
                    {entry.status === "active" ? "Active" : "Archive"} · {entry.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button
            className="download-btn campaign-download-all-btn"
            disabled={downloadingEverything}
            onClick={downloadEverything}
            title="Download every completed campaign trajectory bundle"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            <span>
              {downloadingEverything ? "Downloading..." : "Download Everything"}
            </span>
          </button>
        </div>
        <div className="campaign-stat-row">
          <div className="campaign-stat">
            <span className="campaign-stat-value">{campaign.models.length}</span>
            <span className="campaign-stat-label">Models</span>
          </div>
          {selectedCampaignMeta && (
            <div className="campaign-stat">
              <span className="campaign-stat-value">
                {selectedCampaignMeta.status === "active" ? "active" : "archived"}
              </span>
              <span className="campaign-stat-label">Status</span>
            </div>
          )}
          <div className="campaign-stat">
            <span className="campaign-stat-value">{campaign.task_total}</span>
            <span className="campaign-stat-label">Tasks</span>
          </div>
          {/* Runs Done stat hidden — confusing on campaign tab since it
              counts legacy benchmark dirs, not campaign runs */}
          <div className="campaign-stat">
            <span className="campaign-stat-value">
              {campaign.modes.join(" + ")}
            </span>
            <span className="campaign-stat-label">Modes</span>
          </div>
        </div>
        <div className="campaign-model-chips">
          {campaign.models.map((m) => (
            <span
              key={m}
              className="campaign-model-chip"
              style={{ borderColor: modelColor(m), color: modelColor(m) }}
            >
              {m}
            </span>
          ))}
        </div>
        {selectedCampaignMeta?.archive_reason && (
          <div className="campaign-archive-note">
            Archived: {selectedCampaignMeta.archive_reason}
          </div>
        )}
      </div>

      {/* Panel content */}
      {panel === "leaderboard" && (
        <LeaderboardPanel
          leaderboard={leaderboard}
          campaign={campaign}
          soloRuns={soloRuns}
          matchupRuns={matchupRuns}
          onRunClick={openRun}
          onModelClick={openModelDetail}
        />
      )}
      {panel === "model-detail" && selectedModelDetail && (
        <ModelDetailPanel
          campaign={campaign}
          selection={selectedModelDetail}
          summaryCache={summaryCache}
          onTaskClick={(runKey, taskId) => openTask(runKey, taskId, "model-detail")}
          onTaskDownload={downloadTask}
          downloadingTaskKey={downloadingTaskKey}
        />
      )}
      {panel === "run-detail" && (
        <RunDetailPanel
          runKey={selectedRunKey}
          runDef={campaign.runs[selectedRunKey]}
          summary={runSummary}
          onTaskClick={(taskId) => openTask(selectedRunKey, taskId, "run-detail")}
          onTaskDownload={downloadTask}
          onDownloadAll={downloadRunTasks}
          downloadingRunKey={downloadingRunKey}
          downloadingTaskKey={downloadingTaskKey}
        />
      )}
      {panel === "task-detail" && (
        <div>
          {taskDetail ? (
            <TaskView task={taskDetail} onImageClick={onImageClick} />
          ) : (
            <div className="loading">
              <div className="loading-spinner" />
              Loading task...
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Leaderboard Panel ─── */

function LeaderboardPanel({
  leaderboard,
  campaign,
  soloRuns,
  matchupRuns,
  onRunClick,
  onModelClick,
}: {
  leaderboard: Leaderboard | null;
  campaign: Campaign;
  soloRuns: [string, any][];
  matchupRuns: [string, any][];
  onRunClick: (k: string) => void;
  onModelClick: (model: string, mode: string) => void;
}) {
  const models = campaign.models;
  const modes = campaign.modes;
  const hasSoloData =
    leaderboard && Object.keys(leaderboard.solo).length > 0;
  const hasMatchupData =
    leaderboard && Object.keys(leaderboard.matchups).length > 0;

  return (
    <div className="campaign-panel">
      {/* Solo results matrix */}
      <div className="campaign-section">
        <h3 className="campaign-section-title">Solo Pass Rates <span className="campaign-section-subtitle">cooperative + mixed</span></h3>
        {hasSoloData ? (
          <div className="campaign-table-wrap">
            <table className="campaign-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Mode</th>
                  <th>Evaluated</th>
                  <th>Pass Rate</th>
                  {["cooperative", "mixed"].map((cat) => (
                    <th key={cat}>{cat}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.flatMap((model) =>
                  modes.map((mode) => {
                    const key = `${model}_${mode}`;
                    const entry = leaderboard!.solo[key];
                    return (
                      <tr key={key}>
                        <td>
                          <button
                            className="campaign-model-link"
                            onClick={() => onModelClick(model, mode)}
                          >
                            <span
                              className="campaign-model-dot"
                              style={{ background: modelColor(model) }}
                            />
                            {model}
                          </button>
                        </td>
                        <td className="campaign-mode-cell">{mode}</td>
                        {entry ? (
                          <>
                            <td className="campaign-tasks-cell">
                              <span className="campaign-tasks-done">{entry.overall?.total ?? 0}</span>
                              <span className="campaign-tasks-sep">/</span>
                              <span className="campaign-tasks-total">{campaign.task_total - (campaign.task_counts?.competitive ?? 0)}</span>
                            </td>
                            <td className="campaign-rate-cell">
                              <RateBar rate={entry.overall?.pass_rate ?? entry.pass_rate ?? 0} />
                            </td>
                            {["cooperative", "mixed"].map((cat) => {
                              const cs = entry.categories[cat];
                              return (
                                <td key={cat} className="campaign-rate-cell">
                                  {cs ? (
                                    <RateBar rate={cs.pass_rate} />
                                  ) : (
                                    <span className="campaign-na">--</span>
                                  )}
                                </td>
                              );
                            })}
                          </>
                        ) : (
                          <>
                            <td colSpan={4} className="campaign-pending-cell">
                              pending
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <SoloPreviewGrid
            models={models}
            modes={modes}
            soloRuns={soloRuns}
            onRunClick={onRunClick}
          />
        )}
      </div>

      <div className="campaign-section">
        <h3 className="campaign-section-title">Solo Literal ToM <span className="campaign-section-subtitle">supported end-of-episode probes</span></h3>
        {hasSoloData ? (
          <div className="campaign-table-wrap">
            <table className="campaign-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Mode</th>
                  <th>Probe Tasks</th>
                  <th>Literal ToM</th>
                  {["cooperative", "mixed"].map((cat) => (
                    <th key={cat}>{cat}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.flatMap((model) =>
                  modes.map((mode) => {
                    const key = `${model}_${mode}`;
                    const entry = leaderboard!.solo[key];
                    const overall = entry?.overall;
                    return (
                      <tr key={`literal-${key}`}>
                        <td>
                          <button
                            className="campaign-model-link"
                            onClick={() => onModelClick(model, mode)}
                          >
                            <span
                              className="campaign-model-dot"
                              style={{ background: modelColor(model) }}
                            />
                            {model}
                          </button>
                        </td>
                        <td className="campaign-mode-cell">{mode}</td>
                        {entry ? (
                          <>
                            <td className="campaign-tasks-cell">
                              <span className="campaign-tasks-done">{overall?.literal_tom_task_count ?? 0}</span>
                              <span className="campaign-tasks-sep">/</span>
                              <span className="campaign-tasks-total">{overall?.total ?? 0}</span>
                            </td>
                            <td className="campaign-rate-cell">
                              {typeof overall?.literal_tom_score === "number" ? (
                                <RateBar rate={overall.literal_tom_score} />
                              ) : (
                                <span className="campaign-na">--</span>
                              )}
                            </td>
                            {["cooperative", "mixed"].map((cat) => {
                              const cs = entry.categories[cat];
                              return (
                                <td key={cat} className="campaign-rate-cell">
                                  {typeof cs?.literal_tom_score === "number" ? (
                                    <RateBar rate={cs.literal_tom_score} />
                                  ) : (
                                    <span className="campaign-na">--</span>
                                  )}
                                </td>
                              );
                            })}
                          </>
                        ) : (
                          <td colSpan={4} className="campaign-pending-cell">
                            pending
                          </td>
                        )}
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <SoloPreviewGrid
            models={models}
            modes={modes}
            soloRuns={soloRuns}
            onRunClick={onRunClick}
          />
        )}
      </div>

      {/* Matchup results */}
      <div className="campaign-section">
        <h3 className="campaign-section-title">Competitive Matchups</h3>
        {hasMatchupData ? (
          <div className="campaign-matchup-grid">
            {(() => {
              // Group matchups by model pair
              const grouped: Record<string, { model_a: string; model_b: string; modes: { mode: string; combined: LeaderboardMatchup['combined'] }[] }> = {};
              for (const [, m] of Object.entries(leaderboard!.matchups)) {
                const pairKey = `${m.model_a}_vs_${m.model_b}`;
                if (!grouped[pairKey]) {
                  grouped[pairKey] = { model_a: m.model_a, model_b: m.model_b, modes: [] };
                }
                grouped[pairKey].modes.push({ mode: m.mode || 'unknown', combined: m.combined });
              }
              return Object.entries(grouped).map(([pairKey, group]) => (
                <div key={pairKey} className="campaign-matchup-card">
                  <div className="campaign-matchup-vs">
                    <span
                      className="campaign-matchup-model"
                      style={{ color: modelColor(group.model_a) }}
                    >
                      {group.model_a}
                    </span>
                    <span className="campaign-matchup-vs-label">vs</span>
                    <span
                      className="campaign-matchup-model"
                      style={{ color: modelColor(group.model_b) }}
                    >
                      {group.model_b}
                    </span>
                  </div>
                  {group.modes.map((entry) => {
                    const evaluated = (entry.combined?.model_a_wins ?? 0) + (entry.combined?.model_b_wins ?? 0) + (entry.combined?.draws ?? 0);
                    const compTotal = campaign.task_counts?.competitive ?? 0;
                    return (
                    <div key={entry.mode} className="campaign-matchup-mode-row">
                      <span className="campaign-matchup-mode">{entry.mode}</span>
                      <div className="campaign-matchup-bar-wrap">
                        <MatchupBar
                          aWins={entry.combined?.model_a_wins ?? 0}
                          bWins={entry.combined?.model_b_wins ?? 0}
                          draws={entry.combined?.draws ?? 0}
                          colorA={modelColor(group.model_a)}
                          colorB={modelColor(group.model_b)}
                        />
                      </div>
                      <div className="campaign-matchup-stats">
                        <span>{entry.combined?.model_a_wins ?? 0}W</span>
                        <span>{entry.combined?.draws ?? 0}D</span>
                        <span>{entry.combined?.model_b_wins ?? 0}W</span>
                        <span className="campaign-matchup-total">
                          <span className="campaign-tasks-done">{evaluated}</span>
                          <span className="campaign-tasks-sep">/</span>
                          <span className="campaign-tasks-total">{compTotal}</span>
                        </span>
                      </div>
                    </div>
                    );
                  })}
                </div>
              ));
            })()}
          </div>
        ) : (
          <MatchupPreviewGrid
            matchupRuns={matchupRuns}
            onRunClick={onRunClick}
          />
        )}
      </div>
    </div>
  );
}

/* ─── Solo preview (when no leaderboard data yet) ─── */

function SoloPreviewGrid({
  models,
  modes,
  soloRuns,
  onRunClick,
}: {
  models: string[];
  modes: string[];
  soloRuns: [string, any][];
  onRunClick: (k: string) => void;
}) {
  return (
    <div className="campaign-preview-grid">
      {models.map((model) => (
        <div key={model} className="campaign-preview-model">
          <div className="campaign-preview-model-name">
            <span
              className="campaign-model-dot"
              style={{ background: modelColor(model) }}
            />
            {model}
          </div>
          <div className="campaign-preview-runs">
            {soloRuns
              .filter(([, r]) => r.model === model)
              .map(([key, r]) => (
                <button
                  key={key}
                  className="campaign-run-pill"
                  onClick={() => onRunClick(key)}
                >
                  {statusDot(r.status)}
                  <span>{r.mode}</span>
                  <span className="campaign-run-pill-cat">{r.category}</span>
                </button>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function MatchupPreviewGrid({
  matchupRuns,
  onRunClick,
}: {
  matchupRuns: [string, any][];
  onRunClick: (k: string) => void;
}) {
  // Group by model pair
  const pairs = new Map<string, [string, any][]>();
  for (const entry of matchupRuns) {
    const r = entry[1];
    const pairKey = `${r.model_a} vs ${r.model_b}`;
    const list = pairs.get(pairKey) || [];
    list.push(entry);
    pairs.set(pairKey, list);
  }

  return (
    <div className="campaign-matchup-grid">
      {Array.from(pairs.entries()).map(([pairKey, runs]) => (
        <div key={pairKey} className="campaign-matchup-card">
          <div className="campaign-matchup-vs">
            <span
              className="campaign-matchup-model"
              style={{ color: modelColor(runs[0][1].model_a) }}
            >
              {runs[0][1].model_a}
            </span>
            <span className="campaign-matchup-vs-label">vs</span>
            <span
              className="campaign-matchup-model"
              style={{ color: modelColor(runs[0][1].model_b) }}
            >
              {runs[0][1].model_b}
            </span>
          </div>
          <div className="campaign-matchup-runs">
            {runs.map(([key, r]) => (
              <button
                key={key}
                className="campaign-run-pill"
                onClick={() => onRunClick(key)}
              >
                {statusDot(r.status)}
                <span>{r.mode}</span>
                <span className="campaign-run-pill-cat">
                  {r.direction === "swap" ? "swap" : "fwd"}
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ModelDetailPanel({
  campaign,
  selection,
  summaryCache,
  onTaskClick,
  onTaskDownload,
  downloadingTaskKey,
}: {
  campaign: Campaign;
  selection: ModelDetailSelection;
  summaryCache: SummaryCache;
  onTaskClick: (runKey: string, taskId: string) => void;
  onTaskDownload: (runKey: string, taskId: string) => void;
  downloadingTaskKey: string | null;
}) {
  const relevantRuns = Object.entries(campaign.runs).filter(
    ([, run]) =>
      run.type === "solo" &&
      run.model === selection.model &&
      run.mode === selection.mode,
  );
  const hasPendingSummaries = relevantRuns.some(
    ([runKey, runDef]) =>
      runDef.status === "complete" &&
      summaryCache[`${campaign.campaign_id || "active"}:${runKey}`] === undefined,
  );

  const rows = relevantRuns.flatMap(([runKey, runDef]) => {
    const summary = summaryCache[`${campaign.campaign_id || "active"}:${runKey}`];
    if (!summary) return [];
    return summary.results.map((result) => ({
      runKey,
      runDef,
      result,
    }));
  });

  const groupedRows = rows.reduce<Record<string, typeof rows>>((acc, row) => {
    const category = row.result.category || "unknown";
    if (!acc[category]) acc[category] = [];
    acc[category].push(row);
    return acc;
  }, {});

  return (
    <div className="campaign-panel">
      <div className="campaign-run-detail-header">
        <h3 className="campaign-run-detail-title">
          <span style={{ color: modelColor(selection.model) }}>
            {selection.model}
          </span>
          <span className="campaign-run-detail-sep">/</span>
          {selection.mode}
        </h3>
        <div className="campaign-run-detail-status">
          {rows.length} task{rows.length === 1 ? "" : "s"}
        </div>
      </div>

      {hasPendingSummaries ? (
        <div className="loading">
          <div className="loading-spinner" />
          Loading model tasks...
        </div>
      ) : rows.length > 0 ? (
        Object.entries(groupedRows)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([category, categoryRows]) => (
            <div key={category} className="campaign-section">
              <h3 className="campaign-section-title">
                <span className={`category-badge ${category}`}>{category}</span>
                <span className="campaign-section-subtitle">
                  {categoryRows.length} task
                  {categoryRows.length === 1 ? "" : "s"}
                </span>
              </h3>
              <div className="campaign-task-list">
                {categoryRows.map(({ runKey, result }) => (
                  <div key={`${runKey}:${result.task_id}`} className="campaign-task-row">
                    <button
                      className="campaign-task-row-main"
                      onClick={() => onTaskClick(runKey, result.task_id)}
                    >
                      <span
                        className={`task-status ${result.success ? "success" : "failure"}`}
                      />
                      <span className="campaign-task-row-title">{result.title}</span>
                      <span className="campaign-task-row-meta">
                        {result.turns}t · {result.steps}s
                      </span>
                      {result.evaluation && (
                        <span className="campaign-task-row-progress">
                          {(result.evaluation.percent_complete * 100).toFixed(0)}%
                        </span>
                      )}
                      <span className="campaign-task-row-literal">
                        LitToM {formatLiteralTomScore(literalTomFromEvaluation(result.evaluation))}
                      </span>
                    </button>
                    <button
                      className="download-btn campaign-task-row-download"
                      title="Download this trajectory JSON"
                      disabled={downloadingTaskKey === `${runKey}:${result.task_id}`}
                      onClick={() => onTaskDownload(runKey, result.task_id)}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))
      ) : (
        <div className="campaign-pending-msg">
          No completed tasks are available for this model/mode yet.
        </div>
      )}
    </div>
  );
}

function RunDetailPanel({
  runKey,
  runDef,
  summary,
  onTaskClick,
  onTaskDownload,
  onDownloadAll,
  downloadingRunKey,
  downloadingTaskKey,
}: {
  runKey: string;
  runDef: any;
  summary: CampaignBenchmarkSummary | null;
  onTaskClick: (taskId: string) => void;
  onTaskDownload: (runKey: string, taskId: string) => void;
  onDownloadAll: (
    runKey: string,
    runDef: any,
    summary: CampaignBenchmarkSummary,
  ) => void;
  downloadingRunKey: string | null;
  downloadingTaskKey: string | null;
}) {
  if (!runDef) {
    return <div className="campaign-panel">Run not found: {runKey}</div>;
  }

  const isSolo = runDef.type === "solo";

  return (
    <div className="campaign-panel">
      <div className="campaign-run-detail-header">
        <h3 className="campaign-run-detail-title">
          {isSolo ? (
            <>
              <span style={{ color: modelColor(runDef.model) }}>
                {runDef.model}
              </span>
              <span className="campaign-run-detail-sep">/</span>
              {runDef.mode}
              <span className="campaign-run-detail-sep">/</span>
              {runDef.category}
            </>
          ) : (
            <>
              <span style={{ color: modelColor(runDef.model_a) }}>
                {runDef.team_0}
              </span>
              <span className="campaign-run-detail-vs">vs</span>
              <span style={{ color: modelColor(runDef.model_b) }}>
                {runDef.team_1}
              </span>
              <span className="campaign-run-detail-sep">/</span>
              {runDef.mode}
            </>
          )}
        </h3>
        <div className="campaign-run-detail-actions">
          {summary && (
            <button
              className="download-btn campaign-download-all-btn"
              title="Download all trajectories in this run"
              disabled={downloadingRunKey === runKey}
              onClick={() => onDownloadAll(runKey, runDef, summary)}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              <span>
                {downloadingRunKey === runKey
                  ? "Downloading..."
                  : "Download All"}
              </span>
            </button>
          )}
          <div className="campaign-run-detail-status">
            {statusDot(runDef.status)}
            {runDef.status}
          </div>
        </div>
      </div>

      {summary ? (
        <>
          <div className="campaign-run-stats">
            <div className="campaign-stat">
              <span className="campaign-stat-value">
                {summary.pass_rate.toFixed(1)}%
              </span>
              <span className="campaign-stat-label">Pass Rate</span>
            </div>
            <div className="campaign-stat">
              <span className="campaign-stat-value">
                {formatLiteralTomScore(summary)}
              </span>
              <span className="campaign-stat-label">Literal ToM</span>
            </div>
            <div className="campaign-stat">
              <span className="campaign-stat-value">
                {summary.passed}/{summary.total}
              </span>
              <span className="campaign-stat-label">Passed</span>
            </div>
            <div className="campaign-stat">
              <span className="campaign-stat-value">{summary.failed}</span>
              <span className="campaign-stat-label">Failed</span>
            </div>
          </div>

          {/* Category breakdown */}
          {summary.category_stats && Object.keys(summary.category_stats).length > 0 && (
            <div className="campaign-cat-stats">
              {Object.entries(summary.category_stats).map(([cat, stats]) => (
                <div key={cat} className="campaign-cat-stat">
                  <span className={`category-badge ${cat}`}>{cat}</span>
                  <RateBar rate={stats.pass_rate} />
                  <span className="campaign-cat-stat-detail">
                    {stats.passed}/{stats.total} · avg {stats.avg_steps.toFixed(0)} steps
                    {stats.timed_out > 0 && ` · ${stats.timed_out} timed out`}
                    {typeof stats.literal_tom_score === "number" &&
                      ` · lit ${stats.literal_tom_score.toFixed(0)}%`}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Task results list */}
          <div className="campaign-section">
            <div className="campaign-section-header">
              <h3 className="campaign-section-title">
                Tasks ({summary.results.length})
              </h3>
            </div>
            <div className="campaign-task-list">
              {summary.results.map((r: CampaignRunResult) => (
                <div key={r.task_id} className="campaign-task-row">
                  <button
                    className="campaign-task-row-main"
                    onClick={() => onTaskClick(r.task_id)}
                  >
                    <span
                      className={`task-status ${r.success ? "success" : "failure"}`}
                    />
                    <span className="campaign-task-row-title">{r.title}</span>
                    <span className={`category-badge ${r.category}`}>
                      {r.category}
                    </span>
                    <span className="campaign-task-row-meta">
                      {r.turns}t · {r.steps}s
                    </span>
                    {r.evaluation && (
                      <span className="campaign-task-row-progress">
                        {(r.evaluation.percent_complete * 100).toFixed(0)}%
                      </span>
                    )}
                    <span className="campaign-task-row-literal">
                      LitToM {formatLiteralTomScore(literalTomFromEvaluation(r.evaluation))}
                    </span>
                  </button>
                  <button
                    className="download-btn campaign-task-row-download"
                    title="Download this trajectory JSON"
                    disabled={downloadingTaskKey === `${runKey}:${r.task_id}`}
                    onClick={() => onTaskDownload(runKey, r.task_id)}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : runDef.status === "pending" ? (
        <div className="campaign-pending-msg">
          This run hasn't been executed yet.
          <code className="campaign-empty-cmd">
            ./emtom/run_emtom.sh campaign run --only {runKey}
          </code>
        </div>
      ) : (
        <div className="loading">
          <div className="loading-spinner" />
          Loading run data...
        </div>
      )}
    </div>
  );
}

/* ─── Small components ─── */

function RateBar({ rate }: { rate: number }) {
  const hue = (rate / 100) * 120; // 0=red, 120=green
  const color = `hsl(${hue}, 65%, 45%)`;
  const bgColor = `hsl(${hue}, 50%, 94%)`;
  return (
    <div className="campaign-rate-bar">
      <div className="campaign-rate-bar-track" style={{ background: bgColor }}>
        <div
          className="campaign-rate-bar-fill"
          style={{ width: `${rate}%`, background: color }}
        />
      </div>
      <span className="campaign-rate-bar-label" style={{ color }}>
        {rate.toFixed(0)}%
      </span>
    </div>
  );
}

function MatchupBar({
  aWins,
  bWins,
  draws,
  colorA,
  colorB,
}: {
  aWins: number;
  bWins: number;
  draws: number;
  colorA: string;
  colorB: string;
}) {
  const total = aWins + bWins + draws;
  if (total === 0) return null;
  const aPct = (aWins / total) * 100;
  const dPct = (draws / total) * 100;
  const bPct = (bWins / total) * 100;
  return (
    <div className="campaign-matchup-bar">
      <div style={{ width: `${aPct}%`, background: colorA }} />
      <div style={{ width: `${dPct}%`, background: "var(--border)" }} />
      <div style={{ width: `${bPct}%`, background: colorB }} />
    </div>
  );
}
