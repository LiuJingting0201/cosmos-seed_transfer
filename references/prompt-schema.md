# Cosmos Transfer prompt schema

The generator consumes the prompt JSON verbatim as a string, so the field set matters.
Match the official demo prompt at
`cookbooks/cosmos3/generator/transfer/assets/edge/prompt.json`.

## Field set

Top-level keys, in this order:

```
subjects, background_setting, lighting, aesthetics, cinematography, style_medium,
artistic_style, context, actions, text_and_signage_elements, segments, transitions,
temporal_caption, audio_description, resolution, aspect_ratio, duration, fps
```

Nested:

- `subjects[]` — `description`, `appearance_details`, `relationship`, `location`,
  `relative_size`, `orientation`, `pose`, `action`, `state_changes`, `clothing`,
  `expression`, `gender`, `age`, `skin_tone_and_texture`, `facial_features`,
  `number_of_subjects`, `number_of_arms`, `number_of_legs`
- `lighting` — `conditions`, `direction`, `shadows`, `illumination_effect`
- `aesthetics` — `composition`, `color_scheme`, `mood_atmosphere`, `patterns`
- `cinematography` — `camera_motion`, `framing`, `camera_angle`, `depth_of_field`,
  `focus`, `lens_focal_length`
- `actions[]` — `time`, `description`
- `segments[]` — `segment_index`, `time_range`, `description`, `key_changes`, `camera`

## Rules

- **Delete `scene_imagination`.** The upsampler emits it as a scratchpad and expects the
  deployment layer to strip it, but nothing in the framework does. It is not part of the
  trained schema. Never carry it into `modified_prompt.json` and never put a requirement
  there.
- Include every field. Use an empty string for ones that do not apply (`expression`,
  `age`, `gender` on an inanimate object) rather than omitting them.
- Keep descriptive content concise and grounded. No arbitrary minimum prose length. Retain the native field set, target identity, original robot/gripper appearance and action timing within the deployed tokenizer limit.
- Use the real output width, height, FPS and duration.
- Describe only the finished video. No editing language — no `modify`, `change`,
  `recolor`, `original`, `instead of`, no before/after.
- One exact colour phrase per object, reused everywhere. Never alternate `blue` / `cyan` /
  `sapphire` / `cobalt` for one target.

## Robots and manipulators keep their real appearance

The most common failure is the target colour bleeding onto the robot. The cause is almost
always omission: the prompt says what the floor and the objects should look like and
nothing concrete about the arm, so the model treats it as free surface.

- In each view where they are visibly established, give the robot, arm, gripper and hands
  their own `subjects[]` entry with the full field set. Omit them from views where they
  are not visible; never invent presence merely to preserve a shared identity.
- `appearance_details` must state the **actual source appearance** with a concrete
  material-and-colour phrase read off the input frames — `matte black anodized aluminium
  links with dark grey joint housings`. Not vague filler like `industrial humanoid
  construction with realistic joints`.
- Repeat that phrase in `aesthetics.color_scheme`, so the stated palette contains the
  robot's real colour alongside the target colours. A palette listing only target colours
  leaves the arm unanchored.
- Give cables, fixtures and housings their own stable materials for the same reason, and
  describe accurately any object the user did not ask to restyle.

### The gripper or hand needs its own entry

The end effector is the single most persistent failure in this pipeline. It drifts to
white even when the arm entry already mentions it, because in a wrist view it is not a
detail of the arm — it fills the foreground, it is the largest surface in frame, and it
sits in direct contact with the objects being recoloured. A clause inside the arm's
`appearance_details` does not anchor it.

So the **final active prompt must carry a separate `subjects[]` entry for the gripper or
hand in every view where that gripper or hand is visibly established**, with:

- `description` and `appearance_details` naming its **real source appearance** with a
  concrete material-and-colour phrase read off the frames — `matte black parallel-jaw
  grippers with dark ribbed gripping faces and black pivot hardware`, `white knuckle
  housings with matte black rubber fingertips`;
- an explicit statement that it keeps that colour **in the wrist close-ups where it fills
  the foreground**, and is never tinted by the light or by the colours of the objects it
  holds;
- `state_changes` saying the jaws or fingers open and close but the colour and finish
  never change.

Its source appearance identity belongs to the shared plan. Reuse the exact
`description` and `appearance_details` in every view where the end effector is visible,
but do not inject a subject into a view where it is absent.

## Describe every surface and object that takes up real area

Anything with a meaningful footprint in frame needs an explicit description — whether it
is being restyled or deliberately kept. Not because the prompt should be exhaustive, but
because an undescribed region has no owner, and two things then happen to it:

- a strong nearby description bleeds onto it. A person's hands with no stated appearance,
  sitting next to `matte black grippers that fill the foreground`, come out black.
- each view invents its own answer. A conveyor belt running through a counter, unmentioned
  and invisible in the depth map because it is coplanar with the counter, came out as a
  wooden plank in one camera and unchanged in another.

Depth control makes this sharper: a large flat surface has almost no structure in a depth
map, so its appearance is decided by the prompt alone. Check the control map — anything
that reads as a featureless gradient there is entirely at the mercy of the text.

Practical test before running: list every region covering more than a few percent of the
frame, and confirm each one is named somewhere in the prompt, with either a target
material or an explicit "keeps its own ..." clause.

## Several objects of the same class

Objects sharing a class and a shape and differing only in colour are the fragile case:
colour is their only intrinsic handle.

**Group them by source colour and map each group to one distinct target colour.** Green,
orange and red blocks become sapphire blue, amber yellow and deep violet. Diversity
survives, colour still tells them apart, and the mapping is deterministic.

- The source colour lives only in `transfer_plan.json`, as the key recording the mapping
  (`"green block": "deep sapphire blue"`). It never appears in the prompt — the prompt
  states target colours only, which are unique per group and so work as anchors with no
  circularity and no editing language.
- Target colours must be pairwise distinct **across** source-colour groups, and not
  merely distinct — **far apart in hue**. Neighbouring hues on same-shaped objects are
  the fragile case: sapphire blue and deep violet, about 40 degrees apart, were seen
  swapping between views, while an emerald green in the same scene never drifted once.
  Leave roughly a third of the colour wheel between any two targets in a class — blue /
  green / orange rather than blue / violet / magenta — and prefer differing in lightness
  as well. Within a group the colour is deliberately shared and the members need no
  telling apart; one entry with `number_of_subjects` set to the count is the right
  representation.
- What fails is a lumped phrase spanning different colours, like `three small green,
  orange and red blocks`. It binds nothing. Observed: with that phrasing an orange block
  came out green in every view, leaving two green blocks where the source had one.
