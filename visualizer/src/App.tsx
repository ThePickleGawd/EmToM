import { useState, useEffect, useCallback } from "react";
import type { RunsIndex, TaskDetail } from "./types";
import Sidebar from "./components/Sidebar";
import TaskView from "./components/TaskView";
import CampaignView from "./components/CampaignView";

export type Source = "campaign" | "library";

export default function App() {
  const [runsIndex, setRunsIndex] = useState<RunsIndex | null>(null);
  const [source, setSource] = useState<Source>("campaign");
  const [selectedTaskId, setSelectedTaskId] = useState<string>("");
  const [taskDetail, setTaskDetail] = useState<TaskDetail | null>(null);
  const [loadingTask, setLoadingTask] = useState(false);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/runs.json")
      .then((r) => r.json())
      .then((data: RunsIndex) => {
        setRunsIndex(data);
      });
  }, []);

  const loadTask = useCallback(
    (taskId: string) => {
      setSelectedTaskId(taskId);
      setLoadingTask(true);
      const path = `/data/tasks/_library/${taskId}.json`;
      fetch(path)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then((data: TaskDetail) => {
          setTaskDetail(data);
          setLoadingTask(false);
        })
        .catch((err) => {
          console.error("Failed to load task:", path, err);
          setLoadingTask(false);
        });
    },
    [],
  );

  const handleSourceChange = (s: Source) => {
    setSource(s);
    setSelectedTaskId("");
    setTaskDetail(null);
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightboxSrc(null);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  if (!runsIndex) {
    return (
      <div className="app">
        <div className="loading">
          <div className="loading-spinner" />
          Loading benchmark data...
        </div>
      </div>
    );
  }

  // Campaign mode uses full-width layout (no sidebar)
  if (source === "campaign") {
    return (
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-header">
            <h1>
              <span>EmToM</span> Visualizer
            </h1>
            <div className="source-tabs">
              <button
                className="source-tab active"
                onClick={() => handleSourceChange("campaign")}
              >
                Campaign
              </button>
              <button
                className="source-tab"
                onClick={() => handleSourceChange("library")}
              >
                Library
              </button>
            </div>
          </div>
          <div className="campaign-sidebar-info">
            <p className="campaign-sidebar-hint">
              Campaign results, leaderboard, and competitive matchups.
            </p>
          </div>
        </aside>
        <main className="main-content">
          <CampaignView onImageClick={setLightboxSrc} />
        </main>
        {lightboxSrc && (
          <div
            className="lightbox-overlay"
            onClick={() => setLightboxSrc(null)}
          >
            <img className="lightbox-img" src={lightboxSrc} alt="Frame" />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <Sidebar
        source={source}
        onSourceChange={handleSourceChange}
        libraryTasks={runsIndex.library || []}
        selectedTaskId={selectedTaskId}
        onTaskSelect={loadTask}
      />
      <main className="main-content">
        {!selectedTaskId ? (
          <div className="empty-state">
            Select a task from the sidebar to view its trajectory
          </div>
        ) : loadingTask ? (
          <div className="loading">
            <div className="loading-spinner" />
            Loading task...
          </div>
        ) : taskDetail ? (
          <TaskView task={taskDetail} onImageClick={setLightboxSrc} />
        ) : null}
      </main>

      {lightboxSrc && (
        <div className="lightbox-overlay" onClick={() => setLightboxSrc(null)}>
          <img className="lightbox-img" src={lightboxSrc} alt="Frame" />
        </div>
      )}
    </div>
  );
}
