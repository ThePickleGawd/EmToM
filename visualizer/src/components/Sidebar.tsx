import type { RunSummary } from "../types";

interface Props {
  runs: RunSummary[];
  selectedRunId: string;
  selectedRun?: RunSummary;
  selectedTaskId: string;
  onRunChange: (runId: string) => void;
  onTaskSelect: (taskId: string) => void;
}

export default function Sidebar({
  runs,
  selectedRunId,
  selectedRun,
  selectedTaskId,
  onRunChange,
  onTaskSelect,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>
          <span>EmToM</span> Benchmark Visualizer
        </h1>
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
      </div>
      <div className="task-list">
        {selectedRun?.tasks.map((task) => (
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
