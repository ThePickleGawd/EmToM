import { useState, useEffect, useCallback } from "react";
import type { RunsIndex, TaskDetail, GenerationIndex } from "./types";
import Sidebar from "./components/Sidebar";
import TaskView from "./components/TaskView";
import CampaignView from "./components/CampaignView";
import GenerationView from "./components/GenerationView";

export type Source = "campaign" | "library" | "generation";

export default function App() {
  const [runsIndex, setRunsIndex] = useState<RunsIndex | null>(null);
  const [source, setSource] = useState<Source>("campaign");
  const [selectedTaskId, setSelectedTaskId] = useState<string>("");
  const [taskDetail, setTaskDetail] = useState<TaskDetail | null>(null);
  const [loadingTask, setLoadingTask] = useState(false);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [generationIndex, setGenerationIndex] = useState<GenerationIndex | null>(null);
  const [selectedGenerationId, setSelectedGenerationId] = useState("");

  useEffect(() => {
    fetch("/data/runs.json")
      .then((r) => r.json())
      .then((data: RunsIndex) => {
        setRunsIndex(data);
      });
  }, []);

  // Auto-refresh generation index when on the generation tab
  useEffect(() => {
    if (source !== "generation") return;
    let cancelled = false;

    const loadIndex = () => {
      fetch("/data/generation-index.json")
        .then((r) => r.json())
        .then((data: GenerationIndex) => {
          if (cancelled) return;
          setGenerationIndex(data);
          setSelectedGenerationId((current) => {
            if (data.generations.some((run) => run.id === current)) return current;
            return data.generations[0]?.id || "";
          });
        })
        .catch(() => {
          if (!cancelled) setGenerationIndex({ generations: [] });
        });
    };

    loadIndex();
    const intervalId = window.setInterval(loadIndex, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [source]);

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

  const renderMainContent = () => {
    if (source === "campaign") {
      return (
        <main className="main-content">
          <CampaignView onImageClick={setLightboxSrc} />
        </main>
      );
    }
    if (source === "generation") {
      return (
        <main className="main-content">
          {!selectedGenerationId ? (
            <div className="empty-state">
              {generationIndex && generationIndex.generations.length === 0 ? (
                <div className="campaign-empty">
                  <div className="campaign-empty-icon">&#x2237;</div>
                  <p>No generation runs found.</p>
                  <code className="campaign-empty-cmd">
                    ./emtom/bulk_generate.sh --total-tasks 8 --task-gen-agent mini --model gpt-5.2
                  </code>
                </div>
              ) : (
                "Select a generation run from the sidebar"
              )}
            </div>
          ) : (
            <GenerationView generationId={selectedGenerationId} />
          )}
        </main>
      );
    }
    return (
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
    );
  };

  return (
    <div className="app">
      <Sidebar
        source={source}
        onSourceChange={handleSourceChange}
        libraryTasks={runsIndex.library || []}
        selectedTaskId={selectedTaskId}
        onTaskSelect={loadTask}
        generationIndex={generationIndex}
        selectedGenerationId={selectedGenerationId}
        onGenerationSelect={setSelectedGenerationId}
      />
      {renderMainContent()}
      {lightboxSrc && (
        <div className="lightbox-overlay" onClick={() => setLightboxSrc(null)}>
          <img className="lightbox-img" src={lightboxSrc} alt="Frame" />
        </div>
      )}
    </div>
  );
}
