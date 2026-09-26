import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { App } from "./App";
import { Landing } from "./pages/Landing";
import { LiveRun } from "./pages/LiveRun";
import { RepoList } from "./pages/RepoList";
import { ProjectHealth } from "./pages/ProjectHealth";
import { ProjectCompare } from "./pages/ProjectCompare";
import { RepositoryWorkspace } from "./pages/RepositoryWorkspace";
import "./styles/tokens.css";
import "./styles/app.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Landing /> },
      { path: "app", element: <RepoList /> },
      { path: "app/runs/:runId", element: <LiveRun /> },
      { path: "app/repos/:repositoryId/:section", element: <RepositoryWorkspace /> },
      { path: "app/repos/:repositoryId", element: <RepositoryWorkspace /> },
      { path: "health", element: <ProjectHealth /> },
      { path: "health/compare", element: <ProjectCompare /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
