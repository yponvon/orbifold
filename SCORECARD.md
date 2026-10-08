# Scorecard

This file says where we meet each judging point and where we fall short. The evidence comes from the pixel-error log (`out/progress/LOG.md`), the still frames saved each round (`out/progress/round*_frames/`, REAL on the left and SIM on the right, rows headcam / exocam1 / exocam2), and two independent reviews of rounds 4 and 5 (`out/progress/round4_review.md`, `out/progress/round5_review.md`). The video is `media/compare.mp4`.

> **Round 6 status:** these round-6 changes are built but their effect was not scored when this was written: full-resolution room retrain, cloth-simulated tunic and trousers, window key light, shadow catchers and the closed-loop albedo. The ratings below are based on round 5 and the round-5 review. Update the table once round 6 is scored.

## Judging criteria

| Criterion | Met? | Evidence | Known flaws |
|---|---|---|---|
| **Realism** | Partly | The room looks photographic in the two tripod views. The round-5 review called exocam2 "nearly hard to tell apart". Clothing and hair colours were measured from the footage, and the review confirms they are right. Exocam2 error is 5.9/255. | The person still reads as CG. Round 5 looked flat and cel-shaded with no contact shadows. The iron reads as a slab. The feet came out lime green (the colour sampler picked up the green sheet). A ragged white hem showed at the waist, and the dark collar and sleeve trim is missing. The head-camera view of the room is blurry (half-resolution training) and has a smear under the body. |
| **Fidelity** (pose, timing, camera) | Partly | Every rendered frame uses the tracker pose of that same frame at 30 fps, so the timing is exact by construction. The body fit errors are 1.18 cm (joints) and 0.39 cm (fingertips). The cameras use recorded intrinsics, and the tripod poses come from PnP. The round-5 review says "exo pose, lean and timing match well" and "room edges line up in all three views". | Only 2 s (60 frames) of the 57.6 s recording are rendered. In the head view, the iron is seen edge-on rather than from above, and the legs splay into a V (round-5 review). The fitted pose can only be as good as the dataset's tracking. |
| **Physics** (grip, feet, nothing floats) | Partly | The iron is attached under the right palm on every frame, with its soleplate held flat and never below the bed top. The right fingers curl round it (round-5 review). In round 4 the feet looked grounded in exocam2. The floor height is set 3 cm below the lowest tracked foot. | No contact is simulated. The iron is rigidly attached, not gripped. The left hand hovers above the cloth in the head view (round 5, t = 1.95 s). The trousers being ironed are part of the static room, so the iron does not move or press them. The bed height is a heuristic (lowest left knuckle minus 3 cm). Without contact shadows (round 5), nothing looked grounded. |
| **Honesty** (reproducible, flaws named) | Partly | All code is in `scripts/` and `src/`. Each stage has a Make target and `make all` runs them in order. The README names every stage, dependency and licence, and says the room is learned from the footage. The per-round scores and the independent reviews are kept. | The whole pipeline has not been run from a fresh clone in one go; each stage was run separately during development. The GPU stages need a CUDA machine (we used an RTX 5090). `make test` has no tests to run (there is no `tests/` folder). The dataset is not redistributed and must be downloaded. |

## What to rebuild

| Item | Met? | Evidence | Known flaws |
|---|---|---|---|
| **Cameras** | Yes | Recorded K, 1920×1080, no distortion. `ros_body` axes chosen because 96–100% of joints land in the image, against 0% for the other conventions. Skeleton overlays line up (`out/progress/round0to1_steps/`). The two tripod cameras are each re-solved as one fixed pose, since they are still in the image (0 px shift) while the exocam2 extrinsics drift 0.23 m. Both reviews rate cameras 4/5. | The head-camera pose comes from the recording and is refined only during room training. Small errors there blur the head-camera room render. |
| **Person** (incl. hands) | Partly | Anny body fitted to all tracked body and hand joints: 1.18 cm joints, 0.39 cm fingertips. Fingers are posed from the hand tracking. The garments are cut from the body and draped with cloth simulation (round 6). | Generic face (the dataset face is blurred). The head is hidden for the head camera. The surface had problems in round 5: hem, feet colour, missing trim, flat shading. The left palm is not pinned to the cloth. |
| **Objects touched** | Partly | The iron is modelled from the footage: turquoise base, white shell and handle, steel soleplate, cord. It follows the right palm's yaw on every frame. | The iron looks boxy. The cord is attached to the iron but does not hang to the floor as in the real exocam1 view. The trousers being ironed are not a separate object: they are baked into the room model and stay still. A cloth model of them (stage 03f) was dropped because the camera poses disagree by a few cm between views. |
| **Room** (surfaces, layout, lighting) | Partly | Gaussian-splat room trained on footage with the person masked out. A learned colour response per camera and exposure per image match the cameras' white balance and auto-exposure. An opacity penalty removed floaters (round 3). Both reviews rate the room 4/5. The body is lit by a 360° HDR rendered from this room. | Areas never seen without the person in front (floor and bed under the body in the head view) are smeared guesses. Up to round 5 the model was trained at half resolution, so the head view is blurry; a full-resolution retrain is in progress. The room cannot react to the scene: no moving cloth, and shadows only on proxy shadow catchers. |
| **Render** (same cameras, recording frame rate) | Yes | Blender 5.2 Cycles from all 3 cameras at 1920×1080 and 30 fps, composited over the room render at full resolution. | Only 2 s rendered. The body lighting looked emissive and flat up to round 5. |

## The one rule: "Every pixel must come from your 3D scene"

| Item | Met? | Evidence | Known flaws |
|---|---|---|---|
| No footage pixels, filters or video models over the footage | Yes, with a disclosure | Each output pixel is either rasterised from the 3D Gaussian room or rendered by Cycles from meshes, then alpha-composited. No footage pixels are copied and no image filters or video models are applied. Both reviews check the rule and pass it ("I saw no real-person ghost"). Only scalar colours are measured from the footage for the garment albedo (stage 03c). | **Disclosure:** the room is a 3D model *learned from the same footage* and rendered from the *same viewpoints* it was trained on. Every head-camera frame of the clip was a training image, and the room render reuses that frame's learned pose and exposure. The room therefore matches the footage much more closely than a model built from other views would. The person and iron are never trained on pixels. |

## Known flaws

1. **Short clip.** Only 2 s (60 frames) is rendered, not 15 s and not the full 57.6 s recording.
2. **The room is fitted to the footage it is judged against.** The tripod views and the clip's head-camera frames were training views, so a low room error partly reflects that.
3. **Smear in the head view.** The floor and bed under the body were never seen, so the splat guesses there. It looks like a leak but is not one.
4. **Blurry head-camera room.** Up to round 5 it was trained at half resolution. A full-resolution retrain is in progress.
5. **The trousers being ironed do not move.** They are part of the static room model. The cloth attempt (03f) was dropped because the camera poses disagree by a few cm between views.
6. **No contact physics.** The iron is attached rigidly under the right palm and its soleplate is clamped flat at the bed height. The hand does not really grip it, and the iron does not press the cloth.
7. **Left hand hovers** above the cloth in the head view at times (round-5 review, t = 1.95 s).
8. **Iron angle in the head view.** In round 5 the head camera saw the iron edge-on instead of from above.
9. **The bed height is a heuristic:** the lowest left-knuckle height minus 3 cm. The multi-view solve in 03f is not used in the current scene.
10. **Body surface (round 5):** flat shading, lime-green feet (the sampler picked up the sheet), a ragged white hem at the waist, missing collar and sleeve trim. Round 6 adds a key light, shadow catchers, cloth garments and closed-loop albedo for these, but the effect is not yet scored.
11. **Generic face.** The dataset blurs the real face, so the Anny default face is used. The head is hidden for the head camera, and the real wearer's hair strip at the bottom of that view is only approximated.
12. **Data differs from the brief:** 57.6 s with 2 exo cameras plus a head camera, against the brief's 15 s with 3 exo cameras.
13. **Reproducibility gaps:** the whole pipeline has not been run from a fresh clone in one go. The GPU stages need CUDA. There are no automated tests yet.

## Score history

Mean absolute pixel error between the real and rendered frames, on a 0–255 scale (lower is better). Stage 06 computes it over the 60-frame clip for each camera.

| Round | What changed | headcam | exocam1 | exocam2 |
|---|---|---|---|---|
| 0 | Stick figure, box bed, flat grey room, 960×540 | 124 | 68 | 78 |
| 1 | Room = Gaussian splat (person masked), fixed tripod poses, 1080p | 41 | 14 | 7 |
| 2 | Solid skinned body, head hidden from head camera, tapered iron | 48 | 15 | 7 |
| 3 | Room retrained with opacity penalty, wider body, softer light | 52.3 | 13.6 | 6.5 |
| 4 | Anny body fitted to the tracked joints, dressed by region | 48.9 | 12.4 | 6.8 |
| 5 | Fingertips fitted, colours from footage, 360° room light, iron model flat on bed | 23.0 | 10.0 | 5.9 |
| 6 | Full-res room, cloth garments, key light, shadow catchers *(in progress)* | **TBD** | **TBD** | **TBD** |

Round 0 rendered at 960×540, so its numbers do not compare directly with the later rounds. A lower pixel error does not always mean a more realistic image. For example, the head-camera error rose from round 1 to round 3 while the body became more complete.
