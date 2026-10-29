FROM python:3.11-slim AS runtime

WORKDIR /app
COPY . /app
ENV PYTHON=python

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        make \
        libasound2 \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxcursor1 \
        libxi6 \
        libxinerama1 \
        libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --requirement requirements.lock

CMD ["make", "test"]

# Optional real-browser QA image. The default participant image below stays
# small and does not require a browser, Playwright, or network for make test.
FROM runtime AS ui-qa
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core libfontconfig1 libnspr4 libnss3 libdbus-1-3 \
        libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 \
        libxdamage1 libxfixes3 libgbm1 libxcb1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/qa-browsers
RUN python -m pip install --no-cache-dir --requirement requirements-ui-qa.lock \
    && python -m playwright install chromium --only-shell
CMD ["python", "tools/check_ui_browser.py"]

# Deliberately independent of runtime: no COPY . layer, tracks, configs, runs,
# reports, reward callbacks or organizer inputs ever enter this worker image.
FROM python:3.11-slim AS evaluator
WORKDIR /app
COPY requirements-evaluator.lock /app/requirements-evaluator.lock
RUN python -m pip install --no-cache-dir --requirement requirements-evaluator.lock
COPY racing_event/*.py /app/racing_event/
COPY tools/inference_worker.py /app/tools/inference_worker.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
CMD ["python", "tools/inference_worker.py", "--artifact-dir", "/model"]

FROM runtime AS participant
