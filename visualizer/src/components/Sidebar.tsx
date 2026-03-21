import type { TaskSummary, GenerationIndex } from "../types";
import type { Source } from "../App";

function formatTimestamp(value?: string): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

interface Props {
  source: Source;
  onSourceChange: (s: Source) => void;
  libraryTasks: TaskSummary[];
  selectedTaskId: string;
  onTaskSelect: (taskId: string) => void;
  generationIndex: GenerationIndex | null;
  selectedGenerationId: string;
  onGenerationSelect: (id: string) => void;
}

export default function Sidebar({
  source,
  onSourceChange,
  libraryTasks,
  selectedTaskId,
  onTaskSelect,
  generationIndex,
  selectedGenerationId,
  onGenerationSelect,
}: Props) {
  const renderSidebarBody = () => {
    if (source === "campaign") {
      return (
        <div className="campaign-sidebar-info">
          <p className="campaign-sidebar-hint">
            Campaign results, leaderboard, and competitive matchups.
          </p>
        </div>
      );
    }

    if (source === "generation") {
      if (!generationIndex) {
        return (
          <div className="sidebar-loading">
            <div className="loading-spinner" />
          </div>
        );
      }
      if (generationIndex.generations.length === 0) {
        return (
          <div className="campaign-sidebar-info">
            <p className="campaign-sidebar-hint">
              No generation runs found yet.
            </p>
          </div>
        );
      }
      return (
        <div className="task-list">
          {generationIndex.generations.map((run) => {
            const pct = run.requested_tasks > 0
              ? Math.round((run.submitted_tasks / run.requested_tasks) * 100)
              : 0;
            const label = run.id.replace(/-bulk-generate$/, "");
            return (
              <div
                key={run.id}
                className={`task-item gen-run-item ${selectedGenerationId === run.id ? "active" : ""}`}
                onClick={() => onGenerationSelect(run.id)}
              >
                <div className="task-item-header">
                  <div
                    className={`task-status ${
                      run.submitted_tasks >= run.requested_tasks
                        ? "success"
                        : run.running_workers > 0
                          ? "running"
                          : run.submitted_tasks > 0
                            ? "partial"
                            : "failure"
                    }`}
                  />
                  <span className="task-item-title">{label}</span>
                </div>
                <div className="task-item-meta">
                  <span className="category-badge cooperative">
                    {run.submitted_tasks}/{run.requested_tasks}
                  </span>
                  <span>{run.total_workers}w</span>
                  <span>{formatTimestamp(run.started_at)}</span>
                </div>
                <div className="gen-run-bar">
                  <div
                    className="gen-run-bar-fill"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      );
    }

    // Library
    const total = libraryTasks.length;
    const passed = libraryTasks.filter((t) => t.success).length;
    const passRate = total > 0 ? Math.round((passed / total) * 100) : 0;

    return (
      <>
        {total > 0 && (
          <div className="library-stats">
            <div className="library-stats-rate">{passRate}%</div>
            <div className="library-stats-detail">
              {passed}/{total} passing
            </div>
            <div className="library-stats-bar">
              <div
                className="library-stats-bar-fill"
                style={{ width: `${passRate}%` }}
              />
            </div>
          </div>
        )}
        <div className="task-list">
          {libraryTasks.map((task) => (
            <div
              key={task.task_id}
              className={`task-item ${selectedTaskId === task.task_id ? "active" : ""}`}
              onClick={() => onTaskSelect(task.task_id)}
            >
              <div className="task-item-header">
                <div
                  className={`task-status ${task.success ? "success" : "failure"}`}
                />
                <span className="task-item-title">{task.title}</span>
              </div>
              <div className="task-item-meta">
                {task.category && (
                  <span className={`category-badge ${task.category}`}>
                    {task.category}
                  </span>
                )}
                <span>{task.turns}t</span>
                <span>{task.agents}a</span>
              </div>
            </div>
          ))}
        </div>
      </>
    );
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>
          <span>EmToM</span> Visualizer
        </h1>
        <div className="source-tabs source-tabs-3">
          <button
            className={`source-tab ${source === "campaign" ? "active" : ""}`}
            onClick={() => onSourceChange("campaign")}
          >
            Campaign
          </button>
          <button
            className={`source-tab ${source === "generation" ? "active" : ""}`}
            onClick={() => onSourceChange("generation")}
          >
            Generation
          </button>
          <button
            className={`source-tab ${source === "library" ? "active" : ""}`}
            onClick={() => onSourceChange("library")}
          >
            Library
          </button>
        </div>
      </div>
      {renderSidebarBody()}
    </aside>
  );
}
