import { useState, useEffect, useCallback } from "react";
import type { RunsIndex, RunSummary, TaskDetail } from "./types";
import Sidebar from "./components/Sidebar";
import TaskView from "./components/TaskView";

export type Source = "benchmarks" | "library";

export default function App() {
  const [runsIndex, setRunsIndex] = useState<RunsIndex | null>(null);
  const [source, setSource] = useState<Source>("benchmarks");
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [selectedTaskId, setSelectedTaskId] = useState<string>("");
  const [taskDetail, setTaskDetail] = useState<TaskDetail | null>(null);
  const [loadingTask, setLoadingTask] = useState(false);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/runs.json")
      .then((r) => r.json())
      .then((data: RunsIndex) => {
        setRunsIndex(data);
        if (data.runs.length > 0) {
          setSelectedRunId(data.runs[0].id);
        }
      });
  }, []);

  const selectedRun: RunSummary | undefined = runsIndex?.runs.find(
    (r) => r.id === selectedRunId,
  );

  const loadTask = useCallback(
    (taskId: string) => {
      setSelectedTaskId(taskId);
      setLoadingTask(true);
      const path =
        source === "library"
          ? `/data/tasks/_library/${taskId}.json`
          : `/data/tasks/${selectedRunId}/${taskId}.json`;
      fetch(path)
        .then((r) => r.json())
        .then((data: TaskDetail) => {
          setTaskDetail(data);
          setLoadingTask(false);
        })
        .catch(() => setLoadingTask(false));
    },
    [source, selectedRunId],
  );

  const handleRunChange = (runId: string) => {
    setSelectedRunId(runId);
    setSelectedTaskId("");
    setTaskDetail(null);
  };

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

  return (
    <div className="app">
      <Sidebar
        source={source}
        onSourceChange={handleSourceChange}
        runs={runsIndex.runs}
        libraryTasks={runsIndex.library || []}
        selectedRunId={selectedRunId}
        selectedRun={selectedRun}
        selectedTaskId={selectedTaskId}
        onRunChange={handleRunChange}
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
