import { useEffect, useState } from "react";
import type { TaskDetail, CalibrationMeta } from "../types";
import AgentTimeline from "./AgentTimeline";
import { downloadJson } from "../download";

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

const CALIBRATION_MODE_INFO: Record<string, { label: string; description: string }> = {
  standard: {
    label: "Standard",
    description: "Private secrets only. Agents use the normal benchmark communication and observation channels.",
  },
  baseline: {
    label: "Baseline",
    description: "All secrets are shared. Agents may also read teammates' completed Thought+Action trajectories.",
  },
};

export default function TaskView({ task, onImageClick }: Props) {
  const allAgents = task.llm_agents;
  const tabs = ["all", ...allAgents];
  const [activeTab, setActiveTab] = useState("all");
  const [trajectoryView, setTrajectoryView] = useState<
    "calibration_standard" | "calibration_baseline" | "golden"
  >("calibration_standard");
  const [showPddl, setShowPddl] = useState(false);
  const calibrationByMode = task.calibration_by_mode || {};
  const standardCalibration = calibrationByMode.standard || task.calibration_meta || null;
  const baselineCalibration = calibrationByMode.baseline || null;

  useEffect(() => {
    setTrajectoryView(
      standardCalibration
        ? "calibration_standard"
        : baselineCalibration
          ? "calibration_baseline"
          : "golden",
    );
    setActiveTab("all");
  }, [task.task_id]);

  const selectedCalibration: CalibrationMeta | null =
    trajectoryView === "calibration_baseline"
      ? baselineCalibration
      : standardCalibration;
  const hasGolden = (task.golden_trajectory?.length ?? 0) > 0;
  const hasCalibration =
    (selectedCalibration?.trajectory?.length ?? task.action_history.length) > 0;
  const isLibrary = hasGolden || !!task.task_description;

  const history =
    trajectoryView === "golden" && hasGolden
      ? task.golden_trajectory!
      : selectedCalibration?.trajectory || task.action_history;

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
  const displaySuccess =
    trajectoryView !== "golden" && selectedCalibration
      ? selectedCalibration.passed
      : task.success;
  const displayTurns =
    trajectoryView !== "golden" && selectedCalibration?.turns != null
      ? selectedCalibration.turns
      : task.turns;
  const displaySteps =
    trajectoryView !== "golden" && selectedCalibration?.steps != null
      ? selectedCalibration.steps
      : task.steps;
  const displayResultLabel =
    trajectoryView !== "golden" && selectedCalibration
      ? `${(CALIBRATION_MODE_INFO[selectedCalibration.run_mode]?.label || selectedCalibration.run_mode).toUpperCase()} ${displaySuccess ? "PASSED" : "FAILED"}`
      : displaySuccess
        ? "PASSED"
        : "FAILED";
  const calibrationInfo = CALIBRATION_MODE_INFO[selectedCalibration?.run_mode || "standard"] || {
    label: selectedCalibration?.run_mode || "standard",
    description: "Calibration run mode.",
  };

  return (
    <div>
      <div className="task-header">
        <div className="task-title-row">
          <h2 className="task-title">{task.task_title}</h2>
          <button
            className="download-btn"
            title="Download task JSON"
            onClick={() => downloadJson(task, `${task.task_id}.json`)}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          </button>
        </div>
        <div className="task-meta">
          <span
            className={`task-result-badge ${displaySuccess ? "success" : "failure"}`}
          >
            {displayResultLabel}
          </span>
          <span className="task-meta-chip">{displayTurns} turns</span>
          <span className="task-meta-chip">{displaySteps} steps</span>
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

      {selectedCalibration && (
        <div className="calibration-panel-compact">
          <div className="calibration-meta">
            <span className="calibration-meta-item">
              {selectedCalibration.passed ? "Cal. passed" : "Cal. failed"}
            </span>
            <span className="calibration-meta-item">
              {calibrationInfo.label}
            </span>
            <span className="calibration-meta-item">
              {(selectedCalibration.progress * 100).toFixed(0)}% progress
            </span>
            {selectedCalibration.steps != null && (
              <span className="calibration-meta-item">
                {selectedCalibration.steps} steps
              </span>
            )}
            {selectedCalibration.turns != null && (
              <span className="calibration-meta-item">
                {selectedCalibration.turns} turns
              </span>
            )}
            {Object.entries(selectedCalibration.agent_models).slice(0, 1).map(([, model]) => (
              <span key="model" className="calibration-meta-item">{model}</span>
            ))}
            {selectedCalibration.tested_at && (
              <span className="calibration-meta-item">
                {selectedCalibration.tested_at.slice(0, 16)}
              </span>
            )}
          </div>
          <div className="calibration-mode-note">{calibrationInfo.description}</div>
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

      {(task.literal_tom_probe_results?.length ?? 0) > 0 && (
        <div className="literal-tom-probes">
          <div className="instruction-label">
            Literal ToM Probes
            <span className="literal-tom-summary">
              {task.literal_tom_probe_results!.filter((p) => p.status === "passed").length}
              /{task.literal_tom_probe_results!.filter((p) => p.supported).length} passed
            </span>
          </div>
          {task.literal_tom_probe_results!.map((probe) => (
            <div
              key={probe.probe_id}
              className={`literal-tom-probe ${probe.status}`}
            >
              <div className="probe-header">
                <span className={`probe-status-badge ${probe.status}`}>
                  {probe.status.toUpperCase()}
                </span>
                <span className="probe-agent">{probe.agent_id}</span>
                <code className="probe-source">{probe.source_pddl}</code>
              </div>
              <div className="probe-question">{probe.question}</div>
              {probe.details && (
                <div className="probe-details">
                  <div className="probe-expected">
                    <span className="probe-detail-label">Expected:</span>
                    <code>
                      {probe.details.expected_response.predicate}(
                      {probe.details.expected_response.args.join(", ")}) ={" "}
                      {String(probe.details.expected_response.holds)}
                    </code>
                  </div>
                  <div className="probe-actual">
                    <span className="probe-detail-label">Got:</span>
                    <code>
                      {probe.details.parsed_response.predicate || "(empty)"}
                      {probe.details.parsed_response.args.length > 0
                        ? `(${probe.details.parsed_response.args.join(", ")})`
                        : ""}
                      {probe.details.parsed_response.holds !== null
                        ? ` = ${String(probe.details.parsed_response.holds)}`
                        : ""}
                    </code>
                  </div>
                </div>
              )}
              {probe.raw_response && (
                <details className="probe-raw">
                  <summary>Raw response</summary>
                  <pre>{probe.raw_response}</pre>
                </details>
              )}
            </div>
          ))}
        </div>
      )}

      {isLibrary && (hasGolden || hasCalibration) && (
        <div className="trajectory-toggle">
          {standardCalibration && (
            <button
              className={`trajectory-toggle-btn ${trajectoryView === "calibration_standard" ? "active" : ""}`}
              onClick={() => setTrajectoryView("calibration_standard")}
            >
              Standard Calibration
            </button>
          )}
          {baselineCalibration && (
            <button
              className={`trajectory-toggle-btn ${trajectoryView === "calibration_baseline" ? "active" : ""}`}
              onClick={() => setTrajectoryView("calibration_baseline")}
            >
              Baseline Calibration
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
