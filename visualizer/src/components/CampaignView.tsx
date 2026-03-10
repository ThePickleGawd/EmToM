import { useState, useEffect, useCallback } from "react";
import type {
  Campaign,
  Leaderboard,
  CampaignBenchmarkSummary,
  CampaignRunResult,
  TaskDetail,
} from "../types";
import TaskView from "./TaskView";

interface Props {
  onImageClick: (src: string) => void;
}

type Panel = "leaderboard" | "runs" | "run-detail" | "task-detail";

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

export default function CampaignView({ onImageClick }: Props) {
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [leaderboard, setLeaderboard] = useState<Leaderboard | null>(null);
  const [panel, setPanel] = useState<Panel>("leaderboard");
  const [selectedRunKey, setSelectedRunKey] = useState("");
  const [runSummary, setRunSummary] = useState<CampaignBenchmarkSummary | null>(
    null,
  );
  const [taskDetail, setTaskDetail] = useState<TaskDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/data/campaign.json").then((r) => r.json()),
      fetch("/data/leaderboard.json").then((r) => r.json()),
    ]).then(([c, l]) => {
      setCampaign(c);
      setLeaderboard(l);
      setLoading(false);
    });
  }, []);

  const openRun = useCallback((runKey: string) => {
    setSelectedRunKey(runKey);
    setRunSummary(null);
    setPanel("run-detail");
    fetch(`/data/campaign-run/${runKey}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setRunSummary)
      .catch(() => setRunSummary(null));
  }, []);

  const openTask = useCallback(
    (taskId: string) => {
      setTaskDetail(null);
      setPanel("task-detail");
      fetch(`/data/campaign-task/${selectedRunKey}/${taskId}.json`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then(setTaskDetail)
        .catch(() => setTaskDetail(null));
    },
    [selectedRunKey],
  );

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
          ./emtom/run_emtom.sh campaign create --models gpt-5.2 kimi-k2.5 --modes text vision
        </code>
      </div>
    );
  }

  const runEntries = Object.entries(campaign.runs);
  const soloRuns = runEntries.filter(([, r]) => r.type === "solo");
  const matchupRuns = runEntries.filter(([, r]) => r.type === "matchup");
  const completedCount = runEntries.filter(
    ([, r]) => r.status === "complete",
  ).length;
  const totalCount = runEntries.length;

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
        <button
          className={`campaign-nav-btn ${panel === "runs" ? "active" : ""}`}
          onClick={() => setPanel("runs")}
        >
          All Runs
        </button>
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
              onClick={() => setPanel("run-detail")}
            >
              {selectedRunKey}
            </button>
            <span className="campaign-nav-sep">/</span>
            <span className="campaign-nav-current">Task</span>
          </>
        )}
      </div>

      {/* Campaign header stats */}
      <div className="campaign-header">
        <div className="campaign-stat-row">
          <div className="campaign-stat">
            <span className="campaign-stat-value">{campaign.models.length}</span>
            <span className="campaign-stat-label">Models</span>
          </div>
          <div className="campaign-stat">
            <span className="campaign-stat-value">{campaign.task_total}</span>
            <span className="campaign-stat-label">Tasks</span>
          </div>
          <div className="campaign-stat">
            <span className="campaign-stat-value">
              {completedCount}/{totalCount}
            </span>
            <span className="campaign-stat-label">Runs Done</span>
          </div>
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
      </div>

      {/* Panel content */}
      {panel === "leaderboard" && (
        <LeaderboardPanel
          leaderboard={leaderboard}
          campaign={campaign}
          soloRuns={soloRuns}
          matchupRuns={matchupRuns}
          onRunClick={openRun}
        />
      )}
      {panel === "runs" && (
        <RunsPanel
          soloRuns={soloRuns}
          matchupRuns={matchupRuns}
          onRunClick={openRun}
        />
      )}
      {panel === "run-detail" && (
        <RunDetailPanel
          runKey={selectedRunKey}
          runDef={campaign.runs[selectedRunKey]}
          summary={runSummary}
          onTaskClick={openTask}
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
}: {
  leaderboard: Leaderboard | null;
  campaign: Campaign;
  soloRuns: [string, any][];
  matchupRuns: [string, any][];
  onRunClick: (k: string) => void;
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
        <h3 className="campaign-section-title">Solo Pass Rates</h3>
        {hasSoloData ? (
          <div className="campaign-table-wrap">
            <table className="campaign-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Mode</th>
                  <th>Overall</th>
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
                          <span
                            className="campaign-model-dot"
                            style={{ background: modelColor(model) }}
                          />
                          {model}
                        </td>
                        <td className="campaign-mode-cell">{mode}</td>
                        {entry ? (
                          <>
                            <td className="campaign-rate-cell">
                              <RateBar rate={entry.pass_rate} />
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
                            <td colSpan={3} className="campaign-pending-cell">
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

      {/* Matchup results */}
      <div className="campaign-section">
        <h3 className="campaign-section-title">Competitive Matchups</h3>
        {hasMatchupData ? (
          <div className="campaign-matchup-grid">
            {Object.entries(leaderboard!.matchups).map(([key, m]) => (
              <div key={key} className="campaign-matchup-card">
                <div className="campaign-matchup-vs">
                  <span
                    className="campaign-matchup-model"
                    style={{ color: modelColor(m.model_a) }}
                  >
                    {m.model_a}
                  </span>
                  <span className="campaign-matchup-vs-label">vs</span>
                  <span
                    className="campaign-matchup-model"
                    style={{ color: modelColor(m.model_b) }}
                  >
                    {m.model_b}
                  </span>
                </div>
                <div className="campaign-matchup-bar-wrap">
                  <MatchupBar
                    aWins={m.model_a_wins}
                    bWins={m.model_b_wins}
                    draws={m.draws}
                    colorA={modelColor(m.model_a)}
                    colorB={modelColor(m.model_b)}
                  />
                </div>
                <div className="campaign-matchup-stats">
                  <span>{m.model_a_wins}W</span>
                  <span>{m.draws}D</span>
                  <span>{m.model_b_wins}W</span>
                </div>
              </div>
            ))}
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

/* ─── Runs Panel ─── */

function RunsPanel({
  soloRuns,
  matchupRuns,
  onRunClick,
}: {
  soloRuns: [string, any][];
  matchupRuns: [string, any][];
  onRunClick: (k: string) => void;
}) {
  return (
    <div className="campaign-panel">
      <div className="campaign-section">
        <h3 className="campaign-section-title">
          Solo Runs ({soloRuns.length})
        </h3>
        <div className="campaign-runs-list">
          {soloRuns.map(([key, r]) => (
            <button
              key={key}
              className="campaign-run-row"
              onClick={() => onRunClick(key)}
            >
              {statusDot(r.status)}
              <span
                className="campaign-model-dot"
                style={{ background: modelColor(r.model) }}
              />
              <span className="campaign-run-row-model">{r.model}</span>
              <span className="campaign-run-row-mode">{r.mode}</span>
              <span className={`category-badge ${r.category}`}>
                {r.category}
              </span>
              <span className="campaign-run-row-status">{r.status}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="campaign-section">
        <h3 className="campaign-section-title">
          Matchup Runs ({matchupRuns.length})
        </h3>
        <div className="campaign-runs-list">
          {matchupRuns.map(([key, r]) => (
            <button
              key={key}
              className="campaign-run-row"
              onClick={() => onRunClick(key)}
            >
              {statusDot(r.status)}
              <span
                className="campaign-model-dot"
                style={{ background: modelColor(r.model_a) }}
              />
              <span className="campaign-run-row-model">{r.team_0}</span>
              <span className="campaign-run-row-vs">vs</span>
              <span
                className="campaign-model-dot"
                style={{ background: modelColor(r.model_b) }}
              />
              <span className="campaign-run-row-model">{r.team_1}</span>
              <span className="campaign-run-row-mode">{r.mode}</span>
              <span className="campaign-run-row-status">{r.status}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── Run Detail Panel ─── */

function RunDetailPanel({
  runKey,
  runDef,
  summary,
  onTaskClick,
}: {
  runKey: string;
  runDef: any;
  summary: CampaignBenchmarkSummary | null;
  onTaskClick: (taskId: string) => void;
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
        <div className="campaign-run-detail-status">
          {statusDot(runDef.status)}
          {runDef.status}
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
          {Object.keys(summary.category_stats).length > 0 && (
            <div className="campaign-cat-stats">
              {Object.entries(summary.category_stats).map(([cat, stats]) => (
                <div key={cat} className="campaign-cat-stat">
                  <span className={`category-badge ${cat}`}>{cat}</span>
                  <RateBar rate={stats.pass_rate} />
                  <span className="campaign-cat-stat-detail">
                    {stats.passed}/{stats.total} · avg {stats.avg_steps.toFixed(0)} steps
                    {stats.timed_out > 0 && ` · ${stats.timed_out} timed out`}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Task results list */}
          <div className="campaign-section">
            <h3 className="campaign-section-title">
              Tasks ({summary.results.length})
            </h3>
            <div className="campaign-task-list">
              {summary.results.map((r: CampaignRunResult) => (
                <button
                  key={r.task_id}
                  className="campaign-task-row"
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
                </button>
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
