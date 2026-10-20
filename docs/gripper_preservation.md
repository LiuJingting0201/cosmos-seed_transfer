# Gripper-Preservation Postprocess

The Dex1 Edge+Depth run changes some robot and gripper pixels even though those
objects are outside the requested appearance edit. Segmentation control alone
does not freeze RGB pixels, so the preservation experiment uses segmentation
only to build an alpha mask and then composites source pixels over the generated
video.

## Method

- Head views: SAM3.1 segments both robot arms. The mask builder keeps the two
  largest lower-connected components and fills missed frames with optical flow.
- Wrist views: SAM2.1 tracks the two camera-fixed jaws from positive and negative
  point prompts. A neutral-dark material gate removes adjacent clothing and floor
  pixels.
- Composite: source pixels are resized to the generated resolution and copied
  inside the mask. A 3-pixel feather is applied only at the mask boundary.
- The generated image remains unchanged outside the preservation mask.

## Scripts

- `scripts/build_gripper_preservation_mask.py`
- `scripts/track_wrist_grippers_sam2.py`
- `scripts/composite_preserved_region.py`

Every output has a JSON manifest containing input hashes, frame metadata, mask
coverage, and the output hash. Masks must be visually reviewed before compositing.

## Validated Dex1 Run

- Input generation: Edge `0.6` + Depth `0.4`, seed `2026`
- Views: 4
- Output per view: 300 frames, 10 FPS, 1088x832
- Wrist-right caveat: 49 heavily occluded or low-confidence frames have an empty
  preservation mask and retain the generated pixels.

This is a post-generation 2D preservation step. It does not add 3D or cross-view
geometric constraints.
