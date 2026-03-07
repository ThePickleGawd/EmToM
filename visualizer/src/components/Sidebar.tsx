import type { RunSummary, TaskSummary } from "../types";
import type { Source } from "../App";

interface Props {
  source: Source;
  onSourceChange: (s: Source) => void;
  runs: RunSummary[];
  libraryTasks: TaskSummary[];
  selectedRunId: string;
  selectedRun?: RunSummary;
  selectedTaskId: string;
  onRunChange: (runId: string) => void;
  onTaskSelect: (taskId: string) => void;
}

export default function Sidebar({
  source,
  onSourceChange,
  runs,
  libraryTasks,
  selectedRunId,
  selectedRun,
  selectedTaskId,
  onRunChange,
  onTaskSelect,
}: Props) {
  const tasks = source === "library" ? libraryTasks : selectedRun?.tasks || [];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>
          <span>EmToM</span> Visualizer
        </h1>
        <div className="source-tabs">
          <button
            className={`source-tab ${source === "benchmarks" ? "active" : ""}`}
            onClick={() => onSourceChange("benchmarks")}
          >
            Benchmarks
          </button>
          <button
            className={`source-tab ${source === "library" ? "active" : ""}`}
            onClick={() => onSourceChange("library")}
          >
            Task Library ({libraryTasks.length})
          </button>
        </div>
        {source === "benchmarks" && (
          <>
            <select
              className="run-select"
              value={selectedRunId}
              onChange={(e) => onRunChange(e.target.value)}
            >
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.id}
                </option>
              ))}
            </select>
            {selectedRun && (
              <div className="run-meta">
                {selectedRun.model && (
                  <span className="run-meta-item">{selectedRun.model}</span>
                )}
                <span className="run-meta-item">
                  {selectedRun.passed}/{selectedRun.total} passed
                </span>
                <span className="run-meta-item">
                  {selectedRun.pass_rate.toFixed(0)}%
                </span>
              </div>
            )}
          </>
        )}
      </div>
      <div className="task-list">
        {tasks.map((task) => (
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
    </aside>
  );
}
