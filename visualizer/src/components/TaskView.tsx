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

  const filteredHistory =
    activeTab === "all"
      ? task.action_history
      : task.action_history.filter((a) => a.agent === activeTab);

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
        </div>
      </div>

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
          <div className="instruction-label">Instruction</div>
          <div className="instruction-block">{instructionText}</div>
        </div>
      )}

      <AgentTimeline
        history={filteredHistory}
        agents={allAgents}
        onImageClick={onImageClick}
      />
    </div>
  );
}
