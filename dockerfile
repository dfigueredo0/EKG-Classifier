ARG TORCH_TAG=2.9.0-rocm6.4
ARG BASE_IMAGE=pytorch/pytorch:${TORCH_TAG}

FROM ${BASE_IMAGE} as base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry --version && \
    poetry config virtualenvs.create false  # install into container site-packages

WORKDIR /workspace

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --no-ansi --no-interaction

COPY . .

ENV PYTORCH_HIP_ALLOC_CONF="release_threshold:0.8,max_split_size_mb:128"
ENV HIP_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0

CMD ["bash", "-lc", "poetry run python -m ekgclf.train"]
