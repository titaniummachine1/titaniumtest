FROM rust:1.83-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN git clone https://github.com/titaniummachine1/titanium-ataxx.git /workspace/main-ataxx
COPY . /workspace/titaniumtest
WORKDIR /workspace/titaniumtest
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
RUN RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /workspace/titaniumtest/target/release/titaniumtest /usr/local/bin/titaniumtest
ENTRYPOINT ["titaniumtest"]
CMD ["worker", "--server", "http://host.docker.internal:8080", "--threads", "8"]
