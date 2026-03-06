import type { ActionEntry } from "../types";

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

function extractActionType(action: string): string {
  const match = action.match(/^(\w+)\[/);
  return match ? match[1] : action.split(" ")[0];
}

interface Props {
  history: ActionEntry[];
  agents: string[];
  onImageClick: (src: string) => void;
}

export default function AgentTimeline({ history, agents, onImageClick }: Props) {
  // Group by turn
  const turnGroups = new Map<number, ActionEntry[]>();
  for (const entry of history) {
    const list = turnGroups.get(entry.turn) || [];
    list.push(entry);
    turnGroups.set(entry.turn, list);
  }

  const turns = Array.from(turnGroups.keys()).sort((a, b) => a - b);

  return (
    <div className="timeline">
      {turns.map((turn) => (
        <div key={turn} className="turn-group">
          <div className="turn-label">Turn {turn}</div>
          {turnGroups.get(turn)!.map((entry, i) => {
            const actionType = extractActionType(entry.action);
            const isCommunicate =
              actionType.toLowerCase() === "communicate";
            const color = agentColor(entry.agent, agents);

            return (
              <div
                key={`${turn}-${i}`}
                className={`action-card ${isCommunicate ? "communicate" : ""}`}
              >
                <div className="action-card-header">
                  <span
                    className="agent-dot"
                    style={{ background: color }}
                  />
                  <span className="agent-name" style={{ color }}>
                    {entry.agent}
                  </span>
                  <span className="action-type-badge">{actionType}</span>
                  {entry.skill_steps > 0 && (
                    <span className="skill-steps-badge">
                      {entry.skill_steps} skill steps
                    </span>
                  )}
                </div>
                {entry.thought && (
                  <div className="action-thought">
                    <div className="action-thought-label">Thought</div>
                    {entry.thought}
                  </div>
                )}
                <div className="action-text">{entry.action}</div>
                <div className="action-result">
                  <div className="action-result-label">Result</div>
                  {entry.result}
                </div>
                {entry.frame_paths.length > 0 && (
                  <div>
                    <div className="frame-label">
                      Selected frames ({entry.selected_frames.length})
                    </div>
                    <div className="frame-strip">
                      {entry.frame_paths.map((path, fi) => (
                        <img
                          key={fi}
                          className="frame-thumb"
                          src={path}
                          alt={entry.selected_frames[fi] || `frame-${fi}`}
                          title={entry.selected_frames[fi] || `frame-${fi}`}
                          loading="lazy"
                          onClick={() => onImageClick(path)}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
