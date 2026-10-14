FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv \
    && useradd --create-home --uid 10001 app

COPY pyproject.toml uv.lock ./
COPY README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY dashboard ./dashboard

ENV PATH="/app/.venv/bin:${PATH}"

USER app

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
