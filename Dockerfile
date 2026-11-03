# Live Sky web app. Build: docker build -t orbitguard .   Run: docker run -p 8000:8000 orbitguard
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ORBITGUARD_DATA=/data PORT=8000

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
RUN pip install --no-cache-dir ".[live]"

# Pre-fetch the planetary ephemeris and star catalog so the first sky request isn't slow.
RUN python -c "from orbitguard.live.sky import ephemeris, bright_stars; ephemeris(); bright_stars()"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn orbitguard.live.server:app --host 0.0.0.0 --port ${PORT}"]
