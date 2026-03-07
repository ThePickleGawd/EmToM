import { useState } from "react";
import type { TaskDetail } from "../types";
import AgentTimeline from "./AgentTimeline";

const AGENT_COLORS = [
  "var(--agent-0)",
  "var(--agent-1)",
  "var(--agent-2)",
  "var(--agent-3)",
  "var(--agent-4)",
];

function agentColor(agent: string, agents: string[]): string {
  const idx = agents.indexOf(agent);
  return AGENT_COLORS[idx >= 0 ? idx % AGENT_COLORS.length : 0];
}

interface Props {
  task: TaskDetail;
  onImageClick: (src: string) => void;
}

export default function TaskView({ task, onImageClick }: Props) {
  const allAgents = task.llm_agents;
  const tabs = ["all", ...allAgents];
  const [activeTab, setActiveTab] = useState("all");
  const [trajectoryView, setTrajectoryView] = useState<
    "calibration" | "golden"
  >("calibration");
  const [showPddl, setShowPddl] = useState(false);

  const hasGolden = (task.golden_trajectory?.length ?? 0) > 0;
  const hasCalibration = task.action_history.length > 0;
  const isLibrary = hasGolden || !!task.task_description;

  const history =
    trajectoryView === "golden" && hasGolden
      ? task.golden_trajectory!
      : task.action_history;

  const filteredHistory =
    activeTab === "all"
      ? history
      : history.filter((a) => a.agent === activeTab);

  const instructionText =
    activeTab !== "all" && task.instruction[activeTab]
      ? task.instruction[activeTab]
      : activeTab === "all"
        ? Object.values(task.instruction)[0] || ""
        : "";

  return (
    <div>
      <div className="task-header">
        <h2 className="task-title">{task.task_title}</h2>
        <div className="task-meta">
          <span
            className={`task-result-badge ${task.success ? "success" : "failure"}`}
          >
            {task.success ? "PASSED" : "FAILED"}
          </span>
          <span className="task-meta-chip">{task.turns} turns</span>
          <span className="task-meta-chip">{task.steps} steps</span>
          <span className="task-meta-chip">
            {allAgents.length} agent{allAgents.length !== 1 ? "s" : ""}
          </span>
          {task.mechanics_active.map((m) => (
            <span key={m} className="task-meta-chip">
              {m}
            </span>
          ))}
          {task.tom_level != null && (
            <span className="task-meta-chip">ToM K={task.tom_level}</span>
          )}
        </div>
      </div>

      {task.task_description && (
        <div style={{ marginBottom: 16 }}>
          <div className="instruction-label">Task Description</div>
          <div className="instruction-block">{task.task_description}</div>
        </div>
      )}

      {task.calibration_meta && (
        <div className="calibration-meta">
          <span className="calibration-meta-item">
            {task.calibration_meta.passed ? "Cal. passed" : "Cal. failed"}
          </span>
          <span className="calibration-meta-item">
            {(task.calibration_meta.progress * 100).toFixed(0)}% progress
          </span>
          {Object.entries(task.calibration_meta.agent_models).slice(0, 1).map(([, model]) => (
            <span key="model" className="calibration-meta-item">{model}</span>
          ))}
          {task.calibration_meta.tested_at && (
            <span className="calibration-meta-item">
              {task.calibration_meta.tested_at.slice(0, 16)}
            </span>
          )}
        </div>
      )}

      {task.problem_pddl && (
        <div style={{ marginBottom: 16 }}>
          <button className="pddl-toggle" onClick={() => setShowPddl(!showPddl)}>
            {showPddl ? "Hide" : "Show"} PDDL
          </button>
          {showPddl && (
            <pre className="pddl-block">{task.problem_pddl}</pre>
          )}
        </div>
      )}

      {isLibrary && (hasGolden || hasCalibration) && (
        <div className="trajectory-toggle">
          {hasCalibration && (
            <button
              className={`trajectory-toggle-btn ${trajectoryView === "calibration" ? "active" : ""}`}
              onClick={() => setTrajectoryView("calibration")}
            >
              Calibration Trajectory
            </button>
          )}
          {hasGolden && (
            <button
              className={`trajectory-toggle-btn ${trajectoryView === "golden" ? "active" : ""}`}
              onClick={() => setTrajectoryView("golden")}
            >
              Golden Trajectory
            </button>
          )}
        </div>
      )}

      <div className="agent-tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`agent-tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => setActiveTab(tab)}
            style={
              tab !== "all" && activeTab === tab
                ? { borderBottomColor: agentColor(tab, allAgents) }
                : undefined
            }
          >
            {tab !== "all" && (
              <span
                className="tab-dot"
                style={{ background: agentColor(tab, allAgents) }}
              />
            )}
            {tab === "all" ? "All Agents" : tab}
          </button>
        ))}
      </div>

      {instructionText && activeTab !== "all" && (
        <div>
          <div className="instruction-label">
            {isLibrary ? "Agent Secret" : "Instruction"}
          </div>
          <div className="instruction-block">{instructionText}</div>
        </div>
      )}

      {filteredHistory.length > 0 ? (
        <AgentTimeline
          history={filteredHistory}
          agents={allAgents}
          onImageClick={onImageClick}
        />
      ) : (
        <div className="empty-state" style={{ padding: 40 }}>
          No trajectory data for this view
        </div>
      )}
    </div>
  );
}
