# MeshSplatting (opaque-triangle differentiable renderer) runtime image.
#
# Intentionally no "# syntax=docker/dockerfile:1.7" line: that directive makes
# BuildKit fetch the frontend image from Docker Hub on every build, and on this
# host auth.docker.io resolves to an IPv6 address with no working route. The
# built-in frontend of the current BuildKit understands everything used below
# (heredocs, ARG defaults).
#
# Single-stage CUDA 12.6 devel image: nvcc is required at build time because the
# repository imports three compiled extensions that are not on PyPI:
#   * diff_triangle_rasterization (submodules/diff-triangle-mesh-rasterization)
#   * simple_knn._C               (submodules/simple-knn)
#   * rdel                        (submodules/effrdel)
#
# Build:  docker compose build
# Proof:  docker compose run --rm smoke        (short real training run)
# Docs:   docker/SMOKE.md

# Base image pinned by manifest digest: it makes the build reproducible and lets
# BuildKit reuse the already-pulled layers without re-resolving the tag against
# auth.docker.io (which is unreachable over IPv6 on some networks).
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:d49bb8a4ff97fb5fe477947a3f02aa8c0a53eae77e11f00ec28618a0bcaa2ad1

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
# The PyPI wheel of torch 2.7.1 already targets CUDA 12.6 (it depends on
# nvidia-cuda-runtime-cu12==12.6.77 and nvidia-cudnn-cu12==9.5.1.17), and measured
# throughput on this host was ~8.8 MB/s for files.pythonhosted.org versus ~0.3 MB/s
# for download.pytorch.org and ~0.46 MB/s for mirrors.aliyun.com. PyPI is therefore
# the primary source; the mirrors stay as explicit fallbacks for other networks.
ARG PIP_INDEX_URL_PRIMARY=https://pypi.org/simple
ARG PIP_INDEX_URL_FALLBACK=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_FALLBACK=https://download.pytorch.org/whl/cu126
# RTX 4090 = sm_89. Override for other GPUs, e.g. --build-arg CUDA_ARCH_LIST="8.6;8.9".
ARG CUDA_ARCH_LIST=8.9
ARG MAX_JOBS=8

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------- toolchain
# deadsnakes provides Python 3.11 (README's supported version) on 22.04; the
# distro interpreter is 3.10. libgl1/libglib2.0-0/libgomp1 are runtime needs of
# opencv-python and open3d, both imported at module scope by this repository.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl git \
      build-essential cmake ninja-build pkg-config \
      libgl1 libglib2.0-0 libgomp1 \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv" \
 && rm -rf /var/lib/apt/lists/*

# Put 3.11 first on PATH so `python`/`python3`/`pip` resolve to it (rosdep-style
# apt packages that expect python3.10 are not used at runtime).
RUN ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python \
 && ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python3 \
 && "python${PYTHON_VERSION}" -m ensurepip --upgrade \
 && ln -sf "/usr/local/bin/pip${PYTHON_VERSION}" /usr/local/bin/pip \
 && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -c "import sys; print('[build] python', sys.version.split()[0])"

# --------------------------------------------------------------------- torch
# cu126 wheels match the README (torch 2.7.1 / torchvision 0.22.1, CUDA 12.6).
# pyPI first, then the Tsinghua mirror, then the official PyTorch index; the guard
# after the install fails the build loudly if a non-12.6 build ever gets resolved.
RUN set -eux; \
    python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL_PRIMARY}" \
          "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    || python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL_FALLBACK}" \
          "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    || python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_FALLBACK}" \
          "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    python - <<'PY'
import torch

assert torch.version.cuda is not None, "torch was installed without CUDA support"
assert torch.version.cuda.startswith("12.6"), f"expected a CUDA 12.6 build, got {torch.version.cuda}"
print("[build] torch", torch.__version__, "| cuda", torch.version.cuda)
PY

# ------------------------------------------------- Python runtime dependencies
# requirements.txt in the repository is incomplete (missing open3d, scipy,
# scikit-learn, joblib, mediapy, imageio, scikit-image) and lists packages this
# code never imports (mmcv, mmengine, timm). The container keeps its own list so
# the repository's declared dependency semantics stay unchanged.
COPY docker/requirements-docker.txt docker/constraints-docker.txt /tmp/
# The Ubuntu layer ships distutils-installed copies (blinker 1.4, pyparsing 2.4.7)
# that pip refuses to uninstall, so only blinker is force-overwritten.
RUN python -m pip install --no-cache-dir --ignore-installed --index-url "${PIP_INDEX_URL_PRIMARY}" blinker \
 || python -m pip install --no-cache-dir --ignore-installed --index-url "${PIP_INDEX_URL_FALLBACK}" blinker

# Constraints keep torch/torchvision/numpy at the versions the CUDA extensions are
# compiled against; without them lpips' loose "torch>=0.4.0" would upgrade torch to
# a CUDA 13 build and break the nvcc 12.6 extension build.
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL_PRIMARY}" \
          -c /tmp/constraints-docker.txt -r /tmp/requirements-docker.txt \
 || python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL_FALLBACK}" \
          -c /tmp/constraints-docker.txt -r /tmp/requirements-docker.txt \
 && rm -f /tmp/requirements-docker.txt /tmp/constraints-docker.txt \
 && python -c "import torch; assert torch.version.cuda.startswith('12.6'), torch.version.cuda; print('[build] torch still', torch.__version__, 'cuda', torch.version.cuda)"

# --------------------------------------------------- compiled CUDA extensions
# --no-build-isolation keeps the already-installed torch/numpy/pybind11 in the
# build environment, which is what these setup.py files expect.
COPY submodules /workspace/submodules
WORKDIR /workspace
RUN python -m pip install --no-cache-dir ./submodules/simple-knn --no-build-isolation \
 && python -m pip install --no-cache-dir ./submodules/diff-triangle-mesh-rasterization --no-build-isolation \
 && python -m pip install --no-cache-dir -e ./submodules/effrdel --no-build-isolation

# ------------------------------------------------------- offline model weights
# train.py builds lpips.LPIPS(net='vgg') at startup and metrics.py scores with
# VGG features, so bake the torchvision VGG16 weights (and the LPIPS linear
# weights) into the image and keep runs free of network access.
RUN python - <<'PY'
import torchvision.models as m
import lpips

m.vgg16(weights=m.VGG16_Weights.DEFAULT).eval()
lpips.LPIPS(net="vgg").eval()
print("[build] cached torchvision VGG16 + LPIPS weights")
PY

# ------------------------------------------------------------------ repository
COPY . /workspace
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh docker/smoke.sh \
 && python -c "import triangle_renderer, scene, utils.render_utils; print('[build] repository imports ok')"

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "train.py", "--help"]
