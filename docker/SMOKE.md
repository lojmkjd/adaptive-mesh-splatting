# Docker smoke record

Layered proof that `mesh-splatting:cu126` builds and trains on this host.
Scope is deliberately narrow: **build + one short real training run**. Rendering,
metrics, mesh extraction and segmentation are a follow-up task.

## Environment used for this record

| Item | Value |
|---|---|
| Host | Windows, Docker Desktop 29.8.2, WSL2 backend |
| GPU | NVIDIA GeForce RTX 4090, 24 GiB, driver 617.14 (CUDA UMD 13.4) |
| Base image | `nvidia/cuda:12.6.3-devel-ubuntu22.04` pinned by digest `sha256:d49bb8a4...` |
| Python | 3.11.17 (deadsnakes on Ubuntu 22.04) |
| torch / torchvision | 2.7.1+cu126 / 0.22.1+cu126 (from PyPI) |
| Image | `mesh-splatting:cu126` (`4ab3d80655b7`), 24.5 GB uncompressed |
| Dataset | `D:/workspace/LiGBA-GS/datasets/shudian` (COLMAP `sparse/0`, 28 DJI images, no `images_2`/`images_4`) |

## Commands

```bash
docker compose build                     # cold build: dominated by Docker Hub throughput
docker compose run --rm -T shell python -c "..."       # AC2 import probe
docker compose run --rm -T smoke                       # AC3 short training run
docker compose run --rm -T smoke-offline               # AC4 same run, network disabled
```

Overrides: `DATA_DIR`, `OUTPUT_DIR` (host paths), `SCENE`, `OUT`, `ITERS` (smoke
script), `CUDA_ARCH_LIST` (default `8.9`), `PIP_INDEX_URL_PRIMARY/FALLBACK`
(secondary), `SKIP_GPU_CHECK=1` (bypass the entrypoint GPU probe).

## Results

| Check | Result |
|---|---|
| AC1 build | PASS, `mesh-splatting:cu126`, 24.5 GB (target was ≤20 GB) |
| AC2 imports | `[entrypoint] torch 2.7.1+cu126 / cuda 12.6 / RTX 4090, 22.4 GiB free`; `diff_triangle_rasterization`, `simple_knn._C`, `rdel`, `triangle_renderer`, `scene`, `utils.render_utils` all import |
| AC3 training | PASS: 200 iterations at ~52 it/s on `/data/shudian`, `[smoke] PASS: /outputs/smoke/point_cloud/iteration_200/point_cloud_state_dict.pt (3.5M)` |
| AC4 offline | PASS with `network_mode: "none"`: LPIPS loads from `/usr/local/lib/python3.11/dist-packages/lpips/weights/v0.1/vgg.pth`, VGG16 from the baked torchvision cache |
| AC5 hygiene | PASS: after both runs `git status --short` shows only the new Docker files, no training artifacts inside the repository |

Warm rebuild after a Dockerfile change takes ~23 min on this connection, of which
the VGG16 warm-up (`528 MB` from `download.pytorch.org`) is ~15 min. The cold
build is much longer because Docker Hub base layers run at ~0.4 MB/s.

## Behaviour worth knowing before changing this image

- **`docker compose run <service> <args>` replaces the service command**, it does
  not append. `docker/entrypoint.sh` therefore prepends `python train.py` when the
  first argument is a flag, which is what makes
  `docker compose run --rm train -s /data/scene -m /outputs/run --eval` work.
  Write the full command for anything else: `... train python render.py -m /out`.
- **Git Bash rewrites `/outputs` into `C:/Program Files/Git/outputs`** before the
  argument reaches Docker, which then fails with
  `AssertionError: Could not recognize scene type!`. PowerShell and cmd are not
  affected; in Git Bash prefix the command with `MSYS_NO_PATHCONV=1` (or write
  the paths as `//outputs/...`).
- **Dataset must be writable.** `scene/dataset_readers.py` unconditionally
  creates `<scene>/normals/` next to the image folder and converts
  `sparse/0/points3D.bin` into `points3D.ply` in place on first load. A `:ro`
  mount fails with `OSError: [Errno 30] Read-only file system`. Pre-create both
  artifacts if a read-only mount is required.
- **torch is pinned by constraints, not only by the Dockerfile.** `lpips`
  declares `torch>=0.4.0`; without `docker/constraints-docker.txt` pip upgrades
  torch to a CUDA 13 build and the extension build fails with
  `detected CUDA version (12.6) mismatches ... compiled PyTorch (13.0)`.
- **Only `blinker` is installed with `--ignore-installed`.** The Ubuntu layer
  ships distutils-installed copies that pip refuses to uninstall; applying
  `--ignore-installed` to the whole requirements file re-resolves every
  dependency and silently replaces torch.
- **No `# syntax=` directive.** It makes BuildKit fetch the frontend image from
  Docker Hub on every build; on networks where `auth.docker.io` resolves to an
  unreachable IPv6 address that fails before the first instruction.
- **The base image is pinned by digest** for the same reason: tag resolution goes
  through `auth.docker.io`, digest resolution reuses the local content store.
- **The image is host-CPU specific** because `effrdel` compiles with
  `-march=native`. Rebuild it on the machine that will run it.
- **`CUDA_ARCH_LIST=8.9`** targets sm_89 (RTX 4090). Pass e.g.
  `--build-arg CUDA_ARCH_LIST="8.6;8.9"` for mixed fleets.

## Host troubleshooting

If `docker pull`/`build` hangs on `auth.docker.io` with an IPv6 error
(`dial tcp [2a03:2880:...]:443`), IPv6 is broken on this line. Temporary fix
until Docker Desktop restarts:

```bash
wsl.exe -d docker-desktop -u root sysctl -w \
  net.ipv6.conf.all.disable_ipv6=1 net.ipv6.conf.default.disable_ipv6=1
```

Measured throughput on this host, for choosing sources:

| Source | Speed |
|---|---|
| `files.pythonhosted.org` (PyPI) | ~8.8 MB/s |
| `pypi.tuna.tsinghua.edu.cn` | ~4.7 MB/s |
| `mirrors.aliyun.com/pytorch-wheels` | ~0.46 MB/s |
| `download.pytorch.org` | ~0.29 MB/s |
| Docker Hub | ~0.35–0.6 MB/s |
