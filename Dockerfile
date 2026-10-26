# 统一服务镜像:python 3.11 + babeldoc(v3) + RetainPDF pipeline + typst + 字体。

# ===== Stage 1: 下载 Typst 二进制 + 预置 preview 包(cmarker / mitex)=====
FROM python:3.11-slim-bookworm AS typstsrc

ARG TYPST_VERSION=0.14.2
ARG CMARKER_VERSION=0.1.8
ARG MITEX_VERSION=0.2.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tar xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/typst /opt/typst/bin /opt/typst-packages/preview
RUN set -eux; \
    ARCH="$(uname -m)"; \
    case "$ARCH" in \
      x86_64) TYPST_ARCH="x86_64" ;; \
      aarch64) TYPST_ARCH="aarch64" ;; \
      *) echo "Unsupported architecture: $ARCH"; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-${TYPST_ARCH}-unknown-linux-musl.tar.xz" \
      -o /tmp/typst/typst.tar.xz \
    && tar -xJf /tmp/typst/typst.tar.xz -C /tmp/typst \
    && cp /tmp/typst/typst-${TYPST_ARCH}-unknown-linux-musl/typst /opt/typst/bin/typst

RUN set -eux; \
    for pkg in cmarker:${CMARKER_VERSION} mitex:${MITEX_VERSION}; do \
      name="${pkg%%:*}"; version="${pkg##*:}"; \
      mkdir -p "/tmp/typst/${name}" "/opt/typst-packages/preview/${name}/${version}"; \
      curl -fsSL "https://packages.typst.org/preview/${name}-${version}.tar.gz" \
        -o "/tmp/typst/${name}.tar.gz"; \
      tar -xzf "/tmp/typst/${name}.tar.gz" -C "/tmp/typst/${name}"; \
      cp -R "/tmp/typst/${name}/." "/opt/typst-packages/preview/${name}/${version}/"; \
    done

# ===== Stage 2: 运行时 =====
FROM python:3.11-slim-bookworm AS runtime

# 国内服务器:apt + pip 走清华源
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

# 系统库:v3 侧(pymupdf/opencv/onnxruntime)+ RetainPDF 侧(fontconfig/cjk fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
    ca-certificates curl fontconfig fonts-noto-cjk xz-utils libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv(从 PyPI 清华源)
RUN pip install --no-cache-dir uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# CPU PaddleOCR has its own dependency environment. It does not alter the
# working text-layer translator's NumPy, OpenCV or model dependencies.
COPY local_ocr/requirements.txt /tmp/local-ocr-requirements.txt
RUN uv venv --python /usr/local/bin/python /opt/local-ocr && \
    uv pip install --python /opt/local-ocr/bin/python --no-cache \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/local-ocr-requirements.txt

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    OUTPUT_ROOT=/data \
    PYTHON_BIN=python3 \
    TYPST_BIN=/usr/local/bin/typst \
    TYPST_PACKAGE_PATH=/opt/typst-packages \
    TYPST_PACKAGE_CACHE_PATH=/data/typst-package-cache \
    RETAIN_LOCAL_OCR_BACKEND=embedded \
    RETAIN_LOCAL_OCR_PYTHON=/opt/local-ocr/bin/python \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    PADDLE_PDX_CACHE_HOME=/opt/local-ocr-models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1 \
    RETAIN_PDF_TYPST_FONT_DIRS=/usr/local/share/fonts/pdf-translation \
    RETAIN_PDF_TYPST_FONT_FAMILY=auto

WORKDIR /app

# Typst 二进制 + 预置包
COPY --from=typstsrc /opt/typst/bin/typst /usr/local/bin/typst
COPY --from=typstsrc /opt/typst-packages /opt/typst-packages

# Bundle Latin regular/bold/italic faces and Source Han; register all with Typst.
COPY fonts/ /usr/local/share/fonts/pdf-translation/
COPY fontconfig/65-source-han-serif-alias.conf /etc/fonts/conf.d/65-source-han-serif-alias.conf
RUN fc-cache -f

# babeldoc 依赖(先装 pyproject 声明的依赖,利用层缓存)
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# babeldoc 源码(vendored 包)
COPY babeldoc/ ./babeldoc/

# 服务代码 + RetainPDF pipeline
COPY app.py orchestrate.py classify.py merge.py config.py store.py \
     v3_worker.py v3_models.py retain.py normalize.py run_job.py image_translate.py ./
COPY pipeline ./pipeline
COPY local_ocr ./local_ocr
# Bake common source-language models into the image for local/offline OCR.
ARG LOCAL_OCR_WARMUP_LANGUAGES=en,de,it,zh-CN,zh-TW,ja,tr,ar
RUN /opt/local-ocr/bin/python /app/local_ocr/embedded_paddleocr.py --warmup --languages "$LOCAL_OCR_WARMUP_LANGUAGES"
COPY requirements.txt ./requirements.txt
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN [ -f README.md ] || echo "# unified" > README.md

# 装服务壳依赖 + 可编辑安装 vendored babeldoc + 预热模型/字体
RUN uv pip install --system --no-cache -r requirements.txt
RUN uv pip install --system --no-cache -e /app
# Check the native text-layer parser before publishing the image.
RUN python -c "from babeldoc.format.pdf.new_parser.active_font_backend import ActiveFontFactory; ActiveFontFactory().create_font(None, {'Subtype': 'Type1', 'BaseFont': 'Helvetica', 'Encoding': 'WinAnsiEncoding'})"
RUN babeldoc --warmup

EXPOSE 8040

# 服务端默认配置(env 覆盖)。PORT 由 .env 注入;此处给兜底默认。
ENV WORK_ROOT=/data \
    PORT=8040

ENTRYPOINT ["/entrypoint.sh"]
# 走 app.py 的 __main__:读 settings.port(PORT env),不走写死的命令行 --port
CMD ["python", "app.py"]
