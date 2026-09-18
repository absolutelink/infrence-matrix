FROM python:3.14-slim

ARG S6_OVERLAY_VERSION=3.2.3.0

ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-x86_64.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz

COPY rootfs/ /
COPY additional-services/ /etc/s6-overlay/s6-rc.d/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git cmake \
    && apt-get install -y --no-install-recommends postgresql postgresql-contrib \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash vscode

ENTRYPOINT ["docker-entrypoint"]

CMD ["/init"]