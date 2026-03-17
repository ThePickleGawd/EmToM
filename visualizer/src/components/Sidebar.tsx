import type { TaskSummary } from "../types";
import type { Source } from "../App";

interface Props {
  source: Source;
  onSourceChange: (s: Source) => void;
  libraryTasks: TaskSummary[];
  selectedTaskId: string;
  onTaskSelect: (taskId: string) => void;
}

export default function Sidebar({
  source,
  onSourceChange,
  libraryTasks,
  selectedTaskId,
  onTaskSelect,
}: Props) {
  const tasks = libraryTasks;

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>
          <span>EmToM</span> Visualizer
        </h1>
        <div className="source-tabs">
          <button
            className={`source-tab ${source === "campaign" ? "active" : ""}`}
            onClick={() => onSourceChange("campaign")}
          >
            Campaign
          </button>
          <button
            className={`source-tab ${source === "library" ? "active" : ""}`}
            onClick={() => onSourceChange("library")}
          >
            Library
          </button>
        </div>
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
