# LibreDWG (GPLv3) is built from a pinned release and only invoked as a separate CLI process (dwg2dxf).
# Ubuntu LTS ships no libredwg-tools, and Debian's LibreDWG is older than the DWG formats we want to read.
# Stay on 0.13.3: 0.13.4 writes every layer as switched off (negative colour in the DXF), so previews come out blank.
FROM ubuntu:24.04 AS libredwg
ARG LIBREDWG_VERSION=0.13.3
ARG LIBREDWG_SHA256=83f1f6e78a744777a481ff4520e4cef3f8ac4b2c1c25671077ca12fe81e8816e
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential pkg-config python3 ca-certificates curl xz-utils \
	&& rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o libredwg.tar.xz "https://github.com/LibreDWG/libredwg/releases/download/${LIBREDWG_VERSION}/libredwg-${LIBREDWG_VERSION}.tar.xz" \
	&& echo "${LIBREDWG_SHA256}  libredwg.tar.xz" | sha256sum -c - \
	&& tar -xJf libredwg.tar.xz --strip-components=1 \
	&& ./configure --prefix=/opt/libredwg --disable-shared --enable-static --disable-bindings \
	&& make -j"$(nproc)" \
	&& make install-strip

# Tesseract's accurate ("best") models for German and English (Apache 2.0), pinned to their 4.1.0 release: they read
# the small texts of scanned drawings far better than the fast models of the Ubuntu packages, which they replace.
FROM ubuntu:24.04 AS tessdata
ARG TESSDATA_VERSION=4.1.0
RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates curl \
	&& rm -rf /var/lib/apt/lists/*
WORKDIR /tessdata
RUN for lang in eng deu; do \
		curl -fsSL -o "$lang.traineddata" \
			"https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_VERSION}/$lang.traineddata" || exit 1; \
	done \
	&& printf '%s  %s\n' \
		8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba eng.traineddata \
		8407331d6aa0229dc927685c01a7938fc5a641d1a9524f74838cdac599f0d06e deu.traineddata | sha256sum -c -

FROM ubuntu:24.04 AS python-deps
RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates \
	&& rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# The PP-OCRv6 text detection and recognition models (Apache 2.0) are fetched and checked (SHA-256) by RapidOCR while
# building, into the venv: a conversion never downloads anything.
COPY app/plan2cad/spotting.py /tmp/spotting.py
RUN python3.12 -m venv /opt/venv \
	&& /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
	&& /opt/venv/bin/python -c "import sys; sys.path.insert(0, '/tmp'); import spotting; spotting._engine()"

FROM ubuntu:24.04 AS base
LABEL org.opencontainers.image.title="cad-ingest" \
      org.opencontainers.image.description="Self-hosted CAD conversion service: searchable markdown and PDF/GLB previews of CAD files (DXF, DWG, HPGL, STEP, IGES, BREP, STL, OBJ, PLY, OFF, 3MF, DAE, glTF, IFC), and DXF drawings from photos and scans of drawings" \
      org.opencontainers.image.licenses="Apache-2.0"
# curl: compose/Kubernetes health checks. libgl1: the OpenCascade libraries link against it. fonts: text in drawings.
# tesseract (Apache 2.0, called as a separate process): reads the texts on photos and scans of drawings.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3.12 curl ca-certificates libgl1 fonts-dejavu-core \
		tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng \
	&& rm -rf /var/lib/apt/lists/* \
	&& useradd --system --uid 10001 --no-create-home cad
COPY --from=python-deps /opt/venv /opt/venv
COPY --from=libredwg /opt/libredwg/bin/dwg2dxf /usr/local/bin/dwg2dxf
COPY --from=tessdata /tessdata/ /usr/share/tesseract-ocr/5/tessdata/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /app

# Runs the test suite while building: docker build --target test .
# The tests use a real Redis (started by the tests themselves), installed in this stage only.
FROM base AS test
RUN apt-get update \
	&& apt-get install -y --no-install-recommends redis-server \
	&& rm -rf /var/lib/apt/lists/*
COPY --from=libredwg /opt/libredwg/bin/dwgadd /usr/local/bin/dwgadd
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pytest.ini ./
COPY app app
COPY libraries libraries
COPY bench bench
COPY tests tests
USER 10001
RUN python -m pytest -q

FROM base AS runtime
ARG VERSION=dev
ENV APP_VERSION=$VERSION
COPY LICENSE NOTICE ./
COPY app app
COPY libraries libraries
# One image, two processes: the default command is the API, `worker` runs the RQ workers (see README).
# `cad-ingest` converts a single local file without Redis (python -m app.convert); `cad-ingest library` checks and
# builds symbol libraries. Add your own libraries in /libraries (SYMBOL_LIBRARIES, see docs/symbol-libraries.md).
RUN printf '#!/bin/sh\nexec python -m app.worker "$@"\n' > /usr/local/bin/worker \
	&& printf '#!/bin/sh\nPYTHONPATH=/app exec python -m app.convert "$@"\n' > /usr/local/bin/cad-ingest \
	&& chmod 755 /usr/local/bin/worker /usr/local/bin/cad-ingest
EXPOSE 8080
USER 10001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
