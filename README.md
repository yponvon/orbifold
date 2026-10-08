# orbifold

Real-to-sim rebuild of one recording of a person ironing on a bed. We rebuild the cameras, the room, the person (with hands) and the iron as a 3D scene, render it from the recorded cameras at 30 fps, and play the renders next to the real footage. The room is a Gaussian-splat model trained on footage frames with the person masked out. The person is the Anny parametric body fitted to the tracked 3D joints, dressed in simulated cloth, and rendered in Blender Cycles. The iron is a hand-built mesh. The rendered result is a 2 s clip (60 frames) from all three cameras. [SCORECARD.md](SCORECARD.md) lists which judging points are met and which are not, and names the known flaws.

## Result

Each video shows the ORIGINAL on the left and my render on the right. The rows are headcam, exocam1 and exocam2: 2 s, 60 frames.

| Track | Full resolution (3840×3240) | Web (1920 wide) |
|---|---|---|
| **Track A**: person and iron learned from the footage (mesh-anchored Gaussians, baked iron) | [media/trackA_final.mp4](media/trackA_final.mp4) | [media/trackA_final_web.mp4](media/trackA_final_web.mp4) |
| **Track B**: fully modelled CG person and iron in Blender Cycles | [media/trackB_final.mp4](media/trackB_final.mp4) | [media/trackB_final_web.mp4](media/trackB_final_web.mp4) |

Both tracks share the same calibrated cameras, fitted body and Gaussian-splat room. Earlier rounds and their pixel-error scores are summarised in the scorecard.

## Pipeline

All stages are in `scripts/`. Each script's docstring says what it reads and writes. Outputs go to `out/` (not committed).

| Stage | Make target | What it does |
|---|---|---|
| 00 `00_download.sh` | `data` | Downloads the MCAP from the hackathon Drive folder into `data/`. |
| 01 `01_inspect.py` | `inspect` | Lists topics, calibration, tracker joint names and session labels, and writes a contact sheet. |
| 01b `01b_calibrate.py` | `calib` | Solves one fixed pose per tripod camera with PnP over the whole recording. |
| 02 `02_extract.py` | `extract` | Extracts the clip: frames, per-frame camera poses and world-space joints. |
| 03 `03_overlay.py` | `overlay` | Check: draws the tracked skeleton on the real frames and reports the share of joints inside the image. |
| 03b `03b_fit_body.py` | `fit` | Fits the Anny body (shared shape, per-frame pose) to all tracked joints and fingertips. |
| 03c `03c_sample_colors.py` | `colors` | Measures each garment region's median colour from the footage. Only colours are taken, no pixels. |
| 03d `03d_calibrate_albedo.py` | `albedo` (optional) | Closed-loop albedo correction: compares rendered and real colour per region and rescales. |
| 03e `03e_garments.py` | `garments` | Builds a loose tunic and wide trousers around the body, with pin weights for cloth simulation. |
| 03f `03f_bed_cloth.py` | (not used) | Attempted cloth mesh for the trousers being ironed. Dropped; see limitations. |
| 07 `07_room_data.py` | `room-data` | Room training data: sampled frames, person masks (YOLO + projected skeleton) and camera poses. |
| 08 `08_train_room.py` | `room-train` | Trains the Gaussian-splat room with per-camera colour response, per-image exposure and head-pose refinement. |
| 09 `09_render_room.py` | `room-render` | Renders the room model from each camera for every clip frame. |
| 09b `09b_env_map.py` | `env` | Renders a 360° HDR environment map from the room model at chest height, used to light the body. |
| 04 `04_build_scene.py` | `scene` | Builds the Blender scene: body, garments with cloth sim, iron, shadow catchers, lights and cameras. |
| 05 `05_render.py` | `render` | Renders the foreground (body, iron, shadows; RGBA) from all three cameras in Cycles. |
| 06 `06_compare.py` | `compare` | Composites the foreground over the room render, writes `out/compare.mp4` and prints the per-camera pixel error. |
| 10 `10_color_check.py` | `color-check` (optional) | Compares the lightness of the rendered body with the real person, per garment band. |

The stage numbers do not give the run order. The room (07 to 09b) must be built before the Blender scene (04), because 04 reads the environment map. `make all` runs the stages in the right order.

Helpers: `scripts/preview.sh` rebuilds, renders every 3rd frame on the GPU box and composites. `scripts/save_round.sh` copies a result from the GPU box into `out/progress/`.

## Setup

Python 3.13 managed by [uv](https://docs.astral.sh/uv/). Development was on a Mac. All GPU stages ran on one RTX 5090 (Linux, CUDA 12.8, rented on vast.ai).

```bash
git clone https://github.com/yponvon/orbifold.git && cd orbifold
uv sync                                        # base: MCAP, numpy, OpenCV (enough for 00-03, 03c-03e, 06)
uv sync --extra body --extra recon --extra render   # GPU box: all stages
make data                                      # downloads the MCAP into data/ (or copy it there)
```

Extras (see `pyproject.toml`):
- `render`: `bpy==5.2.2` (Blender 5.2 as a Python module).
- `body`: `torch` (cu128 wheels on Linux) and `anny`.
- `recon`: `torch`, `torchvision`, `gsplat` and `ultralytics`. gsplat compiles its CUDA kernels on first use, so `nvcc` is needed.

The MCAP (`data/ironing_interview.mcap`, 1.1 GB) is not committed. It comes from the [hackathon Drive folder](https://drive.google.com/drive/folders/14GwdeVu9BERb9Y8g4HBIxlNTjtD2R9aB).

## Reproduce

```bash
make all          # calib -> extract -> fit -> colors -> garments -> room-data -> room-train
                  # -> room-render -> env -> scene -> render -> compare
```

You can also run the stages one at a time with the targets above. `CLIP_S` sets the clip length in seconds (default 2, which gives 60 frames). Training the room (`room-train`, 30k steps) takes the longest. Rendering three cameras × 60 frames at 1080p takes a while in Cycles, and `scripts/preview.sh` renders the cameras in parallel for a faster check. To view the scene, open `out/scene.blend` in desktop Blender 5.2.

The whole pipeline has not been run from a fresh clone in one go. Each stage was run on its own while we developed it. See the scorecard (Honesty row).

## Data

What the MCAP contains (checked with stage 01). It differs from the brief.

- **Length:** 57.6 s (1729 frames per camera), not 15 s. The person irons for the whole recording. We render the first 2 s.
- **Cameras:** 1 head camera and 2 exo (tripod) cameras, not 3 exo cameras. The wrist cameras are ignored, as the brief allows.
- **Images:** JPEG, 1920×1080, 30 fps. All topics share timestamps.
- **Intrinsics:** pinhole fx, fy, cx, cy. No distortion fields are set.
- **Extrinsics:** given per frame relative to the pelvis. The pelvis pose is given in world.
- **3D tracking:** 14 upper-body, 8 lower-body and 2×21 hand joints, each with a quality flag.
- **Session labels:** `task_id: ironing`, `tools_involved: [iron]`, `agent_height: 1.75`. There are no interval labels.
- The face is blurred in the dataset.

## Method

### Cameras
- The world pose of each camera is `T_world_cam = T_world_pelvis @ E`, where `E` is the extrinsic (xyz + quaternion xyzw).
- The camera axes are `ros_body` (x forward, y left, z up). We tested each candidate convention by projecting the tracked joints. With `ros_body`, 96–100% of joints land inside the image. With the other conventions, 0% do. The skeleton overlays (stage 03, `out/progress/round0to1_steps/`) confirm this by eye.
- Both tripod cameras are physically still: the image shifts by 0 px over 57 s. The recorded exocam2 extrinsics still drift by about 0.23 m. Stage 01b treats the per-frame projected joints as 2D observations and solves one fixed PnP pose per tripod camera against the world joints from all frames. The head camera keeps its per-frame recorded pose, which the room training refines (below).
- Blender cameras use the recorded focal lengths, principal point and 1920×1080 resolution.

### Room
- The room is a Gaussian-splat model ([gsplat](https://github.com/nerfstudio-project/gsplat)) trained on frames sampled from the whole recording: every 10th tripod frame and every 2nd head-camera frame, plus every head-camera frame inside the rendered clip.
- The person is masked out of the training images. The mask is the union of a YOLO person segmentation and thick 3D capsules around the projected skeleton, which also cover the iron in the hand and the wearer's own body in the head view.
- Learned alongside the splats: a 3×3 + bias colour response per camera, an exposure gain per image (the head camera auto-exposes), and a small pose correction per head-camera image and per tripod camera. An opacity penalty removes floaters in regions no camera sees.
- The model is rendered from the calibrated cameras for every clip frame.
- **Disclosure:** the room is a 3D model learned from the same footage it is compared against, and it is rendered from the same viewpoints. It is a 3D scene (every room pixel is rasterised from 3D Gaussians), but the tripod views fit very closely because those exact viewpoints were in training. Every head-camera frame of the rendered clip was also a training image, and the room render uses that frame's learned pose and exposure. Regions that no camera saw without the person in front of them are guesses. The worst case is the floor and bed under the person in the head view, which comes out as a smear.
- The model used up to round 5 was trained at half resolution, which made the head-camera view blurry. A full-resolution retrain (`SCALE = 1.0` in stage 07) is in progress for round 6.

### Person
- The body is [Anny](https://github.com/naver/anny) (NAVER, Apache-2.0, built on MakeHuman). It needs no licence registration, so we used it in place of SMPL-X.
- The fit (stage 03b) shares one body shape over the clip and has per-bone rotations and a root translation per frame. The loss is the weighted distance from each tracked joint to the matching Anny bone head, including fingertips, plus temporal smoothness and a small pose prior. Tracker quality flags lower the weights.
- Mean fit error: 1.18 cm for the joints and 0.39 cm for the fingertips.
- The face is a generic Anny face because the dataset blurs the real one.
- For the head camera, the head mesh is hidden with a mask modifier, because the camera sits on the head.

### Garments
- Stage 03c projects the fitted body into the tripod views and takes the median real pixel colour per region (skin, shirt, trousers, hair and so on). Only colour values are measured. No footage pixels go into the render. The shoes are skipped because the bed hides the feet in both tripod views.
- The measured colour is divided by the room's mean radiance (from the environment map) to estimate albedo. Stage 03d can then rescale the albedo in a closed loop by comparing rendered and real colour.
- Stage 03e cuts a loose short-sleeved tunic and wide trousers from the body surface. They are pushed outward along the per-frame normals and flare towards the hems, with pin weights (tight at the shoulders and waist, free at the hems). Stage 04 drapes them with Blender's cloth simulation.

### Iron
- The iron is a hand-built mesh about 22 cm long: a turquoise base, a white shell and handle, a steel soleplate and a cord. It is attached rigidly under the right palm.
- Its yaw comes from a palm frame (from the wrist to the mean of the index, middle and pinky knuckles). It only rotates about the vertical axis, so the soleplate stays flat. It is never placed below the bed top.
- The bed-top height used for this is the lowest point of the left middle knuckle in the clip minus 3 cm, because the left hand rests flat on the cloth. Stage 03f also solves the bed height across views, but 03f is not part of the current scene.

### Lighting
- Stage 09b renders a cube map from the room model at the person's chest height and turns it into a 360° equirectangular HDR. That HDR is the Blender world light, so the room's window and ceiling light the body.
- A soft sun light ("window key", strength 0.6) points along the mean direction of the brightest 0.5% of the environment map (the window). It adds shading and shadows.
- The bed, back wall and floor are invisible shadow-catcher proxies. The body and iron cast shadows onto them, and those shadows are composited onto the room render.

### Render and comparison
- Stage 05 renders the foreground in Blender 5.2 Cycles on the GPU (OptiX or CUDA, whichever it finds first) with GPU denoising and a transparent background (RGBA), at the recorded resolution and 30 fps.
- Stage 06 alpha-composites the foreground over the room render for each camera and frame. It writes the side-by-side video and prints the mean absolute pixel error (0–255) per camera, plus the worst frame.
- No footage pixels, filters or video models are applied to the output.

## Credits and licences

| Component | Licence | Use |
|---|---|---|
| [Anny](https://github.com/naver/anny) (NAVER) | Apache-2.0 | Parametric body model |
| MakeHuman assets used by Anny | CC0 | Body mesh and topology |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | Gaussian-splat rasteriser for the room |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`yolo11x-seg`) | AGPL-3.0 | Only for person masks in room training data (07) and the colour check (10). It is not part of the rendered scene. |
| [Blender](https://www.blender.org/) / `bpy` 5.2.2 | GPL | Scene, cloth simulation, Cycles rendering |
| PyTorch, OpenCV, NumPy, mcap | BSD-3 / Apache-2.0 / BSD-3 / MIT | Fitting, image I/O, MCAP decoding |

Dataset: the hackathon recording `ironing_interview.mcap`, which is not redistributed here.
