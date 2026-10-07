# orbifold

Real-to-sim reconstruction of a real recording of someone ironing. The recording is rebuilt as a 3D scene, rendered from the recorded cameras. Every pixel comes from the 3D scene.

Status: work in progress. See [PLAN.md](PLAN.md) for the plan and timeline, and `SCORECARD.md` (added at the end) for which judging points are met.

## Setup
```bash
git clone https://github.com/yponvon/orbifold.git && cd orbifold
uv sync                 # Mac / CPU: data tools only
uv sync --extra gpu     # 5090 box: + torch, smplx, bpy
make data               # downloads the MCAP into data/ (or copy it there)
make inspect            # stage 01: topics, calibration, trackers, contact sheet
```
The MCAP (`data/ironing_interview.mcap`, 1.1 GB) is not committed. It comes from the [hackathon Drive folder](https://drive.google.com/drive/folders/14GwdeVu9BERb9Y8g4HBIxlNTjtD2R9aB).
