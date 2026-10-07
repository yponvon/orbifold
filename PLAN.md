# Orbifold — Real-to-Sim Hackathon Plan (5 hours)

> **Editing:** this is the living plan. Edit it freely. Claude re-reads it before each stage, and your edits win.

## Context
The challenge: rebuild a real recording of someone ironing as a 3D scene whose renders are hard to tell apart from the footage. The hard rule is that **every pixel must come from our 3D scene**.

**Judged on:**
- **Realism**
- **Fidelity:** pose, timing and camera match frame by frame
- **Physics:** grip, feet on the floor, nothing floats
- **Honesty:** reproducible, flaws named

**Deliverables in `github.com/yponvon/orbifold`:** the code, a side-by-side video, a README describing the method and a scorecard.

**Constraints:**
- 5 hours in total.
- Mac M4 Pro: no CUDA, about 11 GB free disk.
- RTX 5090 Linux box reached over SSH.

**Strategy:** get a complete but crude pipeline working end to end on a 5 s clip by about hour 3. Then fix the worst frames, then write up. A finished, honest submission beats a half-built realistic one.

## What the MCAP actually contains (inspected `ironing_interview.mcap`, 1.13 GB)
The file differs from the brief. **It is 57.6 s long, not 15 s**, so we must find the ironing window. It has **2 exo cams, not 3**, and it already includes tracking data.

| Topic(s) | Rate | Use |
|---|---|---|
| `/headcam/image`, `/exocam1/image`, `/exocam2/image` (`recording.protos.Image`, codec JPEG/PNG) | 30 Hz, 1729 frames each | The 3 cameras we render |
| `/{cam}/intrinsics` (`SensorCalibration` → `PinholeCamera`: fx, fy, cx, cy, w, h + Brown-Conrady **or** Kannala-Brandt distortion) | once | Camera lens |
| `/{cam}/extrinsics` (`Transform`: xyz + quaternion xyzw, source → destination frame) | 30 Hz | **Head-camera pose is given per frame.** No need to estimate ego motion |
| `/upperbody/tracking`, `/lowerbody/tracking`, `/hand/tracking/{left,right}` (`Trackers`: named `Pose`s + status) | 30 Hz | **3D body and hand joints are given.** We fit SMPL-X to these directly |
| `/upperbody/exo_view_tracking`, `/lowerbody/exo_view_tracking` | 30 Hz | The same in the exo frame. Cross-check |
| `/session` (labels + interval annotations), `/sync`, `/file_metadata` | — | `/session` may mark the ironing interval |
| `/left_wrist_cam`, `/right_wrist_cam` | — | Ignored, as the brief allows |

- Chunks are **zstd**-compressed. The protobuf schemas are embedded, so decoding uses `mcap` + `mcap-protobuf-support`.

### Stage 01 findings (verified 2026-10-07, `make inspect`)
- **Images:** JPEG, 1920×1080, RGB. 1729 frames per camera, all timestamps aligned (same log_time across topics).
- **Lenses:** pure pinhole, **no distortion fields set** (cx, cy = image centre). The distortion-remap step is dropped. Focal lengths: exocam1 fx=987.2, exocam2 fx=871.5, headcam fx=1121.6.
- **Frames:** `/upperbody/tracking` `pelvis` is the only pose in `world`. Every other joint and all 3 camera extrinsics are given in the `pelvis` frame.
- **Camera convention (tested):** `T_world_cam = T_world_pelvis @ E`, where E is the extrinsic xyz + quaternion xyzw. With this, exocam1 stays fixed in world (std 3 mm in x/y). Exocam2 drifts by about 0.23 m std in x even though it's on a tripod, so the pelvis world pose probably drifts. Use per-frame poses, and smooth or fix the exo cams if the reprojection test shows jitter. Camera axis convention (OpenCV vs OpenGL) is settled by the reprojection test.
- **Body trackers (all pelvis-relative):**
  - upperbody (14): pelvis, spine1-3, neck, head, collar/shoulder/elbow/wrist ×2.
  - lowerbody (8): hip/knee/ankle/foot ×2.
  - hands (21 each): wrist + thumb/index/middle/ring/pinky cmc|mcp/pip/dip/tip.
  - These are SMPL-X joint names, so the fit maps them 1:1.
  - Each joint has an `is_quality_bad` flag. Use it as a fit weight.
- **Session:** `task_id: ironing`, `agent_height: 1.75` (prior for SMPL-X shape), `tools_involved: [iron]`. No interval labels. **Ironing runs the whole 57.6 s**, so the dev clip is 0–5 s, then 0–15 s.
- **Scene (from frames):** ironing on a **bed with a bright green sheet** (not an ironing board): navy trousers and red/white clothes on top, light-blue/white iron with a cord. The room has white walls with a flower decal, a grey sofa, a tiled floor and a projector screen. The headcam image looks **upside down** (mounted inverted), which the reprojection test will confirm.

## Decisions (and why)
| Area | Choice | Why |
|---|---|---|
| Where code runs | Edit on the Mac and push to GitHub. On the 5090, `git pull` and `rsync` the MCAP over once, then run there | The MCAP is already on the Mac. Frames and renders stay on the 5090 (Mac disk) |
| MCAP in git | **Never committed.** Add `*.mcap` to `.gitignore` first; it lives in `data/` | 1.1 GB, GitHub caps files at 100 MB, no git-lfs. Reproduce via README's Drive link + `scripts/00_download.sh` (gdown) |
| Python env | `uv` project, Python **3.11**, `uv.lock` committed | Reproducible. The `bpy` wheel needs 3.11 |
| Window | Find the ironing interval from `/session` labels (or by eye from a contact sheet). Dev clip = 5 s inside it, then 15 s | The file is 57.6 s |
| Cameras | K from intrinsics (pinhole, no distortion). Per-frame `T_world_pelvis @ E` | Verified in stage 01 |
| Person | **Fit SMPL-X (`smplx`, PyTorch) to the given 3D body + hand trackers**, with temporal smoothing and foot-floor contact. Fallback: `rtmlib` 2D keypoints on 2 exo views + triangulation | The trackers save about an hour of keypoint work and are more accurate |
| Objects | Bed = box + mattress + green sheet plane. Iron = a mesh **parented to the right-hand joint** with a fixed grip offset. Cloth = a static textured plane | Guaranteed grip. Cloth sim is named as a flaw |
| Room | Blender boxes/planes for floor, walls and table. One area light + world light matched by eye. *Stretch:* Gaussian-splat background | Fast and controllable |
| Renderer | `bpy` headless **Cycles + OptiX**, low samples + denoiser | Exact intrinsics, fast on the 5090 |
| Comparison | `ffmpeg` hstack of real and render per camera, plus a 2×3 grid. Per-frame error metric | Demo video + "worst frame" selection |
| Quality | `ruff` + `pre-commit`. `pytest` checks: project the tracker joints into each camera and they land on the person (reprojection) | Catches frame/axis bugs, the #1 time sink |
| Git | Small commits on `main` per stage, pushed often | Recoverable. The 5090 always pulls the latest |

## Repo layout
```
orbifold/
  PLAN.md  README.md  SCORECARD.md
  pyproject.toml  uv.lock  .gitignore  .pre-commit-config.yaml  Makefile
  configs/clip.yaml        # mcap path, topics, clip start/end, cameras
  src/orbifold/
    mcap_io.py             # decode protobuf msgs by topic/time; JPEG → ndarray
    cameras.py             # intrinsics/extrinsics → K, dist, T_world_cam; distortion remap
    trackers.py            # tracker poses → joint arrays in world frame
    fit_smplx.py           # SMPL-X fit to trackers (+ smoothing, foot contact)
    scene.py               # bpy: room, board, iron-on-hand, lights, cameras, body anim
    render.py              # per-camera Cycles render at 30 fps + distortion
    compare.py             # side-by-side / grid videos, per-frame error
  scripts/ 00_download.sh 01_inspect.py 02_extract.py 03_fit_body.py 04_build_scene.py 05_render.py 06_compare.py
  tests/test_reprojection.py
  data/ (gitignored: the .mcap)   out/ (gitignored: frames, fits, renders)
```
`make all CLIP=5s` runs stages 01 to 06. Each stage caches to `out/<stage>/`.

## Timeline (commit and push after each stage)
| Time | Milestone | Done when |
|---|---|---|
| 0:00–0:20 | **Setup.** gitignore, move the MCAP to `data/`, PLAN.md, scaffold, `uv sync` on both machines, rsync the MCAP to the 5090. **Register for SMPL-X now** (smpl-x.is.tue.mpg.de) | `01_inspect.py` prints tracker joint names, frame ids, one decoded image, and the `/session` labels |
| 0:20–0:50 | **Cameras.** Clip 0–5 s: extract 150 frames × 3 cams. Build cameras | `pytest`: tracker joints project onto the person in all 3 views |
| 0:50–1:50 | **Body.** SMPL-X fit to the trackers | Mesh overlay matches in all views. Feet don't slide or sink |
| 1:50–2:40 | **Objects + room** in Blender | A still render from exocam1 lines up with the real frame |
| 2:40–3:20 | **Animate + render** 3 cams × 150 frames, with distortion | `out/render/<cam>/%04d.png` |
| 3:20–4:20 | **Compare, fix the worst frame, repeat.** Scale to 15 s if stable, or try the splat stretch | Side-by-side video + error plot |
| 4:20–5:00 | **Ship.** README, SCORECARD, final video, clean-clone `make all` check, push | All deliverables present. **No new features after 4:20** |

If a stage overruns by more than 15 min, take its fallback and move on:
- **Body:** if the trackers are too sparse, use rtmlib + triangulation, or HMR2 on the best view.
- **Room:** flat-coloured boxes.

## Risks to resolve early
- **SMPL-X license download needs an account.** Do it at minute 0.
- **Tracker frames:** the trackers may be in a different world frame than the camera extrinsics. The reprojection test at 0:50 settles it.
- **Disk:** the Mac has about 11 GB free. Extract frames only on the 5090.
- **Honesty:** no footage textures on the people or objects. All approximations go in the README.

## Verification
1. `pytest`: tracker-joint reprojection error below a threshold in all 3 cameras.
2. Mesh overlays on real frames at the clip's start, middle and end, checked by eye.
3. A physics check prints the min foot height vs floor, iron–hand distance (constant) and any object below the floor.
4. `compare.py` writes side-by-side MP4s and a per-frame error. The worst frame is named in SCORECARD.
5. Fresh clone on the 5090 → `uv sync` → `make all CLIP=5s` reproduces the renders.

## First actions once approved
1. Add `.gitignore` (`*.mcap`, `data/`, `out/`, `.venv/`) and move `ironing_interview.mcap` → `data/`. Nothing large gets staged.
2. Copy this plan to `orbifold/PLAN.md`, scaffold the layout and `pyproject.toml`, then commit and push.
3. Write `01_inspect.py`. You run `uv sync && uv run scripts/01_inspect.py` (the sandbox here can't reach PyPI), then paste the output back, and we build stage 02 on it.
