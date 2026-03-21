import { useEffect, useMemo, useState } from "react";
import type {
  AgentTraceEntry,
  GenerationDetail,
  GenerationEvent,
  GenerationWorker,
} from "../types";

function formatTimestamp(value?: string): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function statusLabel(status: GenerationWorker["status"]): string {
  return status.replace("_", " ");
}

function EventPill({ event }: { event: GenerationEvent }) {
  const label =
    event.command ||
    event.event_type.replace(/_/g, " ");
  return (
    <div className={`generation-event-pill ${event.success === false ? "failure" : ""}`}>
      <div className="generation-event-pill-head">
        <span>{label}</span>
        <time>{formatTimestamp(event.timestamp)}</time>
      </div>
      {event.message && <div className="generation-event-pill-copy">{event.message}</div>}
      {event.error && <div className="generation-event-pill-copy failure">{event.error}</div>}
      {event.output_path && (
        <div className="generation-event-pill-copy mono">{event.output_path}</div>
      )}
    </div>
  );
}

function AgentTraceCard({ entry }: { entry: AgentTraceEntry }) {
  if (entry.kind === "tool_call") {
    return (
      <div className="agent-trace-card tool-call">
        <div className="agent-trace-kicker">Command</div>
        <pre>{entry.command || entry.tool}</pre>
      </div>
    );
  }
  if (entry.kind === "tool_result") {
    return (
      <div className="agent-trace-card tool-result">
        <div className="agent-trace-kicker">
          Tool Result
          {typeof entry.returncode === "number" && (
            <span className="agent-trace-returncode">rc={entry.returncode}</span>
          )}
        </div>
        <pre>{entry.output || "(no output)"}</pre>
      </div>
    );
  }
  return (
    <div className="agent-trace-card assistant">
      <div className="agent-trace-kicker">Assistant</div>
      <pre>{entry.content || "(empty)"}</pre>
    </div>
  );
}

function SuccessGraph({ detail }: { detail: GenerationDetail }) {
  const width = 760;
  const height = 240;
  const padding = 28;
  const series = detail.success_series;

  const points = series.map((point, index) => {
    const x =
      series.length <= 1
        ? width / 2
        : padding + (index / (series.length - 1)) * (width - padding * 2);
    const y = height - padding - point.cumulative_pass_rate * (height - padding * 2);
    return { ...point, x, y };
  });

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  return (
    <div className="gen-section">
      <div className="gen-section-head">
        <div>
          <div className="gen-kicker">Calibration Curve</div>
          <h3>Cumulative success rate over submitted tasks</h3>
        </div>
        <div className="gen-stat-pills">
          <span>{detail.submitted_tasks} submitted</span>
          <span>
            {series.length > 0
              ? `${Math.round(series[series.length - 1].cumulative_pass_rate * 100)}% pass`
              : "No submissions"}
          </span>
        </div>
      </div>
      <svg
        className="gen-graph"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Cumulative success rate graph"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y = height - padding - tick * (height - padding * 2);
          return (
            <g key={tick}>
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                className="gen-grid-line"
              />
              <text x={10} y={y + 4} className="gen-axis-label">
                {Math.round(tick * 100)}%
              </text>
            </g>
          );
        })}
        {points.length > 0 && (
          <>
            <path d={path} className="gen-graph-line" />
            {points.map((point) => (
              <g key={`${point.task_id}-${point.index}`}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={6}
                  className={point.success ? "gen-point success" : "gen-point failure"}
                />
                <title>
                  {`${point.index}. ${point.title} \u00b7 ${point.success ? "pass" : "fail"} \u00b7 ${Math.round(point.cumulative_pass_rate * 100)}% cumulative`}
                </title>
              </g>
            ))}
          </>
        )}
      </svg>
      <div className="gen-submission-strip">
        {series.map((point) => (
          <div key={`${point.worker_id}-${point.task_id}`} className="gen-submission-chip">
            <span className={`gen-submission-dot ${point.success ? "success" : "failure"}`} />
            <div>
              <div>{point.title}</div>
              <div className="gen-submission-meta">
                #{point.index} &middot; {point.category} &middot; {point.worker_id}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

interface Props {
  generationId: string;
}

export default function GenerationView({ generationId }: Props) {
  const [detail, setDetail] = useState<GenerationDetail | null>(null);
  const [selectedWorkerId, setSelectedWorkerId] = useState("");
  const [lastUpdatedAt, setLastUpdatedAt] = useState("");

  useEffect(() => {
    if (!generationId) return;

    let cancelled = false;

    const loadDetail = () => {
      fetch(`/data/generation/${encodeURIComponent(generationId)}.json`)
        .then((r) => r.json())
        .then((data: GenerationDetail) => {
          if (cancelled) return;
          setDetail(data);
          setSelectedWorkerId((current) =>
            data.workers.some((worker) => worker.id === current)
              ? current
              : data.workers[0]?.id || "",
          );
          setLastUpdatedAt(new Date().toISOString());
        })
        .catch(() => {
          if (!cancelled) {
            setDetail(null);
          }
        });
    };

    loadDetail();
    const intervalId = window.setInterval(loadDetail, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [generationId]);

  const selectedWorker = useMemo(
    () => detail?.workers.find((worker) => worker.id === selectedWorkerId) || detail?.workers[0] || null,
    [detail, selectedWorkerId],
  );

  if (!detail) {
    return (
      <div className="loading">
        <div className="loading-spinner" />
        Loading generation detail...
      </div>
    );
  }

  return (
    <div className="gen-view">
      {/* Header / metrics bar */}
      <div className="gen-header">
        <div className="gen-header-left">
          <div className="gen-kicker">Generation Run</div>
          <h1 className="gen-title">{detail.id.replace(/-bulk-generate$/, "")}</h1>
          <p className="gen-subtitle">
            {detail.categories.join(" \u00b7 ")} &middot; {detail.requested_tasks} tasks requested
          </p>
        </div>
        <div className="gen-metrics-row">
          <div className="gen-metric-pill">
            <span className="gen-metric-label">Requested</span>
            <span className="gen-metric-value">{detail.requested_tasks}</span>
          </div>
          <div className="gen-metric-pill">
            <span className="gen-metric-label">Submitted</span>
            <span className="gen-metric-value">{detail.submitted_tasks}</span>
          </div>
          <div className="gen-metric-pill">
            <span className="gen-metric-label">Finished</span>
            <span className="gen-metric-value">{detail.finished_workers}</span>
          </div>
          <div className="gen-metric-pill">
            <span className="gen-metric-label">Failed</span>
            <span className="gen-metric-value">{detail.failed_workers}</span>
          </div>
        </div>
      </div>

      <div className="gen-live-tag">
        <span className="gen-live-dot" /> live &middot; refreshes 3s &middot; {formatTimestamp(lastUpdatedAt)}
      </div>

      <SuccessGraph detail={detail} />

      {/* Worker picker */}
      <div className="gen-section">
        <div className="gen-kicker">Workers</div>
        <div className="gen-worker-picker">
          {detail.workers.map((worker) => (
            <button
              key={worker.id}
              className={`gen-worker-tile ${selectedWorker?.id === worker.id ? "active" : ""}`}
              onClick={() => setSelectedWorkerId(worker.id)}
              title={`${worker.id}\nGPU ${worker.gpu}:${worker.slot} · ${worker.category} · K=${worker.current_k_level ?? "?"}\n${worker.submitted_count}/${worker.target_tasks} submitted${worker.fail_reason ? `\n${worker.fail_reason}` : ""}`}
            >
              <span className={`gen-tile-status-bar ${worker.status}`} />
              <span className="gen-tile-category">{worker.category.slice(0, 4)}</span>
              <span className="gen-tile-progress">
                {worker.submitted_count}/{worker.target_tasks}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Selected worker detail */}
      {selectedWorker && (
        <div className="gen-detail-columns">
          <div className="gen-detail-col">
            <div className="gen-section">
              <div className="gen-section-head">
                <div>
                  <div className="gen-kicker">Worker Detail</div>
                  <h3>{selectedWorker.id}</h3>
                </div>
                <span className={`gen-worker-status ${selectedWorker.status}`}>
                  {statusLabel(selectedWorker.status)}
                </span>
              </div>
              <div className="gen-stat-pills" style={{ marginTop: 10 }}>
                <span>{selectedWorker.task_gen_agent || "agent"} / {selectedWorker.task_gen_model || "unknown"}</span>
                <span>{selectedWorker.workspace_id}</span>
                <span>{selectedWorker.scene_id || "no scene"}</span>
              </div>
              {selectedWorker.fail_reason && (
                <div className="gen-fail-banner">{selectedWorker.fail_reason}</div>
              )}
              <div className="gen-item-list">
                {selectedWorker.submitted_tasks.length === 0 ? (
                  <div className="gen-empty-copy">No submitted tasks yet.</div>
                ) : (
                  selectedWorker.submitted_tasks.map((task) => (
                    <div key={task.task_id} className="gen-task-row">
                      <div>
                        <div>{task.title}</div>
                        <div className="gen-submission-meta">
                          {task.category} &middot; {task.task_id}
                        </div>
                      </div>
                      <div className={`gen-badge ${task.success ? "success" : "failure"}`}>
                        {task.success ? "pass" : "fail"}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="gen-section">
              <div className="gen-section-head">
                <div>
                  <div className="gen-kicker">EmToM Events</div>
                  <h3>Taskgen milestones</h3>
                </div>
              </div>
              <div className="gen-item-list">
                {selectedWorker.events.length === 0 ? (
                  <div className="gen-empty-copy">
                    This workspace predates event logging or emitted no events.
                  </div>
                ) : (
                  selectedWorker.events.map((event, index) => (
                    <EventPill key={`${event.timestamp}-${index}`} event={event} />
                  ))
                )}
              </div>
            </div>
          </div>

          <div className="gen-detail-col">
            <div className="gen-section gen-transcript">
              <div className="gen-section-head">
                <div>
                  <div className="gen-kicker">Agent Transcript</div>
                  <h3>{selectedWorker.task_gen_agent === "mini" ? "mini-swe-agent trace" : "agent activity"}</h3>
                </div>
                {selectedWorker.agent_stats && (
                  <div className="gen-stat-pills">
                    {selectedWorker.agent_stats.api_calls != null && (
                      <span>{selectedWorker.agent_stats.api_calls} API calls</span>
                    )}
                    {selectedWorker.agent_stats.instance_cost != null && (
                      <span>${selectedWorker.agent_stats.instance_cost.toFixed(2)}</span>
                    )}
                  </div>
                )}
              </div>
              <div className="gen-item-list">
                {selectedWorker.agent_trace.length === 0 ? (
                  <div className="gen-empty-copy">No structured agent transcript found for this worker.</div>
                ) : (
                  selectedWorker.agent_trace.map((entry) => (
                    <AgentTraceCard key={`${entry.kind}-${entry.index}-${entry.command || entry.content || ""}`} entry={entry} />
                  ))
                )}
              </div>
            </div>

            <div className="gen-section">
              <div className="gen-section-head">
                <div>
                  <div className="gen-kicker">Log Excerpt</div>
                  <h3>Worker stdout/stderr</h3>
                </div>
              </div>
              <div className="gen-log-block">
                <pre>{selectedWorker.log_excerpt.tail.join("\n")}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
