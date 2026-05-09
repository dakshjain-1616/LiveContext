# LiveContext

## Goal
Build a real-time streaming context window monitor for LLM agents that intercepts API calls (OpenAI, Anthropic, Ollama), parses context snapshots, detects evictions, calculates attention (via embeddings), and visualizes the state in a production-ready React dashboard.

## Research Summary
- **Model IDs & Limits (April 2026)**:
    - `gpt-4o`: 128,000 tokens.
    - `claude-3-5-sonnet-20241022`: 200,000 tokens.
    - `gemini-2.0-flash`: 1,000,000 tokens.
    - `llama3.3`: 128,000 tokens.
    - `gemma-3`: 128,000 tokens (for 4B/12B/27B).
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers) for attention estimation.
- **Tech Stack**: FastAPI (Backend), React + Vite + Tailwind + Framer Motion (Frontend), SQLite (Storage), WebSockets (Real-time).

## Approach
- **Backend**: FastAPI app serving as both the dashboard API and a proxy server. Proxy logic will intercept requests, extract message history, calculate tokens/embeddings, and broadcast snapshots.
- **Storage**: SQLite for session persistence and embedding cache.
- **Frontend**: React SPA with WebSocket integration. Components will use Framer Motion for smooth transitions as context shifts.
- **SDK**: A simple Python class `ContextMonitor` for direct integration without the proxy.

## Subtasks
1. **Initialize Project Structure**: Create directory layout and base config files (pyproject.toml, requirements.txt).
2. **Data Models**: Implement Pydantic models for `ContextSnapshot`, `ContextMessage`, and `Eviction`.
3. **Storage Layer**: Implement SQLite schema and `db.py` for persistence and embedding caching.
4. **Core Logic (Capture)**: Implement `tokenizer.py`, `embedder.py` (MiniLM), and `eviction.py` (hash + cosine similarity).
5. **Proxy Interceptors**: Implement OpenAI, Anthropic, and Ollama proxy handlers with streaming support.
6. **Backend Server**: Implement FastAPI app, WebSocket manager, and static file serving for frontend.
7. **Frontend Foundation**: Set up Vite + React + Tailwind + Lucide.
8. **Frontend Components**: Build `ContextStream`, `TokenGauge`, `TimelineScrubber`, and `SessionSelector`.
9. **SDK & CLI**: Implement `monitor.py` SDK and `cli.py` using Click.
10. **Demo & Testing**: Create `demo.py` for synthetic data and write pytest suite.
11. **Final Integration**: Build frontend and verify end-to-end with `livecontext serve`.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/home/daksh/7May/projects/livecontext/server/main.py` | Main FastAPI entry point |
| `/home/daksh/7May/projects/livecontext/server/proxy/` | Interceptor logic |
| `/home/daksh/7May/projects/livecontext/frontend/` | React source code |
| `/home/daksh/7May/projects/livecontext/monitor.py` | Python SDK |
| `/home/daksh/7May/projects/livecontext/demo.py` | Synthetic data generator |

## Evaluation Criteria
- **Real-time**: Snapshots appear in dashboard < 100ms after proxy interception.
- **Accuracy**: Token counts match `tiktoken` for OpenAI models.
- **Functionality**: Eviction detection correctly identifies removed messages.
- **UI/UX**: Smooth animations and clear visualization of attention/token usage.
- **Robustness**: Proxy forwards requests even if monitoring logic fails (fail-safe).
