FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# The container runs as uid 1000 so files it writes into the mounted ./data stay
# owned by the host user. Give that uid a real passwd entry and a home: the
# bundled Claude CLI reads its login from $HOME/.claude, and an unwritable HOME
# (the "/" a uid with no entry gets) stalls it.
RUN useradd --uid 1000 --create-home --shell /bin/bash app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p /app/data /home/app/.claude && chown -R 1000:1000 /app/data /home/app

# CLAUDE_CONFIG_DIR keeps the CLI's whole state — login *and* config — inside the
# one directory docker-compose mounts, instead of half of it in $HOME/.claude.json.
USER app
ENV HOME=/home/app \
    CLAUDE_CONFIG_DIR=/home/app/.claude

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false"]
