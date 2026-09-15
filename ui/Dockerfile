FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app
# ``streamlit run ui/app.py`` puts ``/app/ui`` on sys.path, while the shared
# map module is intentionally imported as ``ui.map_view``. Keep the repository
# root importable in the container as it already is during local development.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN pip install --no-cache-dir pandas==2.2.3 pydeck==0.9.1 streamlit==1.53.0 httpx==0.28.1 streamlit-js-eval==1.0.0
COPY ui ./ui
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
