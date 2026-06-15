# ML Workshop Dashboard (Frontend)

This is the Next.js frontend dashboard for the AI/ML Workshop, providing a modern, real-time graphical interface to control, configure, and monitor machine learning tasks.

## Features

- **Interactive Configuration**: Renders dynamic configuration forms derived directly from backend Pydantic models/JSON schemas.
- **Real-Time Progress Streaming**: Connects via Server-Sent Events (SSE) to display task progress steps and metrics live.
- **Interactive Metric Visualizations**: Features dynamic charts rendering live metrics (such as loss or accuracy curves) during training runs.
- **Cooperative Task Cancellation**: Send interruption signals to cancel running or pending tasks directly from the UI.

## Getting Started

### 1. Requirements

Ensure the FastAPI backend is running. By default, the frontend is configured to proxy API requests to `http://localhost:8000`.

### 2. Install Dependencies

Run the following command in the `frontend/` directory to install dependencies:

```bash
npm install
```

### 3. Run the Development Server

Start the Next.js development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to view the dashboard.

## Development Scripts

- `npm run dev` – Starts the development server.
- `npm run build` – Builds the production application.
- `npm run start` – Starts the production server.
- `npm run lint` – Runs ESLint on the frontend codebase.
