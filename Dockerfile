# Runs every camera of a site (run-site). Works on CPU, or an NVIDIA GPU when started
# with GPU access: PyTorch's Linux wheels include CUDA. All models are in the image, so a
# site needs no internet. The face-age model for --age is left out: it has no licence for
# commercial use.
FROM python:3.12-slim

# OpenCV's shared libraries (its FFmpeg for RTSP comes inside the opencv-python wheel), and
# time zones, so zone "hours" follow the site's TZ instead of UTC.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen && uv run --frozen download-models --skip-age

RUN useradd --create-home --uid 1000 app && mkdir -p logs status && chown -R app /app
USER app
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["uv", "run", "--frozen", "--offline"]
CMD ["run-site", "site.toml"]
