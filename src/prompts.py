MOTION_DIGESTING_TEMPLATE = """You are converting numeric camera-motion statistics of a <CLIP_SECONDS>-second video clip into a clear camera-motion caption.

For fine-grained description, the whole clip it is split into non-overlapping motion units, each spanning <UNIT_SECONDS> seconds. You receive statistics for the overall motion and for each unit. Interpret the numbers and naturally describe the camera motion as a whole in a way that a later video-captioning VLM can use to understand how the camera moves through time.

Return JSON only. Do not include Markdown.

Camera-coordinate convention:
- The poses are camera-to-world matrices.
- Relative motion is expressed in the local camera coordinate system of the starting pose.
- +X means camera-right.
- +Y means camera-down.
- +Z means camera-forward.
- All translations and rotations describe camera motion, not object motion.

Camera motion overview:
- "overall_motion" summarizes the full clip from the first sampled pose to the last sampled pose. Use it to understand the dominant global displacement, rotation, duration, and whether the path is mostly direct or indirect.
- "motion_units" is a chronological list of fixed-duration temporal segments. Use it to preserve phase order, direction changes, brief holds, and changes in motion strength over time.
- Fields beginning with "unit_start_in_clip_start_" describe the where that unit fits in the full clip. Use them to connect units into one continuous motion.
- Fields beginning with "unit_end_in_unit_start_" describe what happens inside that unit. Use them as the main evidence for that unit's local motion.
- Unless stated otherwise, time is measured in seconds, translation is measured in meters, and rotation is measured in degrees.

Overall motion fields:
- "clip_duration_(s)": clip duration, in seconds.
- "clip_start_to_clip_end_path_length_(m)": accumulated camera path length from clip start to clip end. It is non-negative and can exceed the chord length.
- "clip_start_to_clip_end_chord_length_(m)": straight-line camera displacement between the clip start and clip end.
- "clip_end_in_clip_start_right_(m)": net camera displacement along camera-right.
- "clip_end_in_clip_start_down_(m)": net camera displacement along camera-down.
- "clip_end_in_clip_start_forward_(m)": net camera displacement along camera-forward.
- "clip_end_in_clip_start_yaw_(deg)": net yaw change. Positive means pan or turn right; negative means pan or turn left.
- "clip_end_in_clip_start_pitch_(deg)": net pitch change. Positive means tilt up; negative means tilt down.
- "clip_end_in_clip_start_roll_(deg)": net roll change. Positive means clockwise roll; negative means counterclockwise roll.

Motion-unit fields:
- "unit_index": zero-based unit index.
- "unit_begins_at_clip_(s)" and "unit_duration_(s)": when the unit begins and how long it lasts.
- "unit_start_in_clip_start_right_(m)", "unit_start_in_clip_start_down_(m)", "unit_start_in_clip_start_forward_(m)": where this unit begins relative to the first pose of the full clip.
- "unit_start_in_clip_start_yaw_(deg)", "unit_start_in_clip_start_pitch_(deg)", "unit_start_in_clip_start_roll_(deg)": the unit-start orientation relative to the first pose of the full clip.
- "unit_start_to_unit_end_path_length_(m)" and "unit_start_to_unit_end_chord_length_(m)": accumulated and straight-line translation within the unit.
- "unit_end_in_unit_start_right_(m)", "unit_end_in_unit_start_down_(m)", "unit_end_in_unit_start_forward_(m)": unit-local net translation.
- "unit_end_in_unit_start_yaw_(deg)", "unit_end_in_unit_start_pitch_(deg)", "unit_end_in_unit_start_roll_(deg)": unit-local net rotation.

Candidate camera-motion vocabulary:
- move forward, move backward, push in, pull back, dolly in, dolly out,
- approach, retreat from, move toward, move away from,
- slide left, slide right, truck left, truck right, track left, track right,
- strafe left, strafe right, drift left, drift right, glide left, glide right,
- rise, lower, move up, move down, ascend, descend, pedestal up, pedestal down,
- crane up, crane down,
- pan left, pan right, turn left, turn right, rotate left, rotate right,
- pivot left, pivot right, sweep left, sweep right,
- tilt up, tilt down,
- roll left, roll right, bank left, bank right,
- arc left, arc right, curve left, curve right, veer left, veer right,
- orbit, partial orbit, move around, circle partially around,
- pause, hold, settle, remain still, stay steady, static, nearly still,
- slowly, smoothly, gradually, gently, steadily, quickly, abruptly, slightly, strongly, sharply.

Signed-value interpretation:
- Right translation: positive means slide/truck/track right; negative means slide/truck/track left.
- Down translation: positive means lower or move down; negative means rise or move up.
- Forward translation: positive means move forward, push in, or approach; negative means move backward, pull back, or retreat.
- Yaw delta: positive means pan or turn right; negative means pan or turn left.
- Pitch delta: positive means tilt up; negative means tilt down.
- Roll delta: positive means roll clockwise; negative means roll counterclockwise.

Core output principle:
- The caption should be spatially imaginable and temporally grounded.
- Use numeric values when they are meaningful for understanding the dominant motion, especially total duration, major translations, and major rotations.
- Do not force every small number into the text.
- Prefer natural rounded quantities over raw precision.
- Avoid vague motion words when they hide important numeric evidence.
- The final motion caption should mention the dominant global displacement or rotation quantitatively when the numbers are nontrivial.
- Tiny motion components should not compete with dominant ones. Mention small components only when they mark a real phase change or disambiguate direction.

Motion interpretation guidance:
- Use unit-local fields to describe motion within each unit.
- Use unit-start-in-clip-start fields only to understand how units connect into a continuous path.
- If accumulated path is much larger than net displacement, interpret conservatively as curved, indirect, wavered, back-and-forth, or shaky according to the per-unit sequence.
- Meaningful translation plus meaningful yaw may be described as an arc, curve, sweep, or veer.
- Forward motion plus lateral motion may be described as a diagonal move or curved approach.
- Lateral motion plus yaw in the same general direction may be described as an arc or sweeping move.
- Lateral motion plus yaw in the opposite direction may indicate tracking while looking back or panning across the scene.
- Very small translation and very small rotation should be described as mostly static, nearly still, or a brief hold.
- Pose-derived forward/backward motion is physical camera movement, not optical zoom.
- Use "orbit" only when the trajectory strongly suggests curved motion around a center.
- Prefer simple motion descriptions when the evidence is ambiguous.
- Be conservative with strong terms such as "sharp", "abrupt", "quick", "orbit", or "large turn" unless the numbers clearly support them.

Numeric wording policy:
- Always preserve temporal order.
- Each motion-unit description must include its time range.
- Include meters or degrees only for dominant, meaningful, or disambiguating motion.
- Omit or qualitatively summarize tiny components that do not affect the perceived camera path.
- If translation is below about 0.05 m and rotation is below about 2 degrees, describe the unit as nearly still unless other evidence says otherwise.
- If a unit has a dominant component above about 0.10 m or 5 degrees, include a rounded quantity for that component in the unit description.
- If the full clip has net translation above about 0.20 m, path length above about 0.30 m, or rotation above about 10 degrees, include the rounded scale in the final motion caption.
- Round times to about 0.1 seconds.
- Round translations naturally: for small but meaningful motion, use about 0.05 m or 0.1 m precision; for larger motion, use about 0.1 m or coarser.
- Round rotations naturally: usually to the nearest 1, 5, 10, or 15 degrees depending on scale.
- Avoid raw-looking values such as 0.237 m or 13.842 degrees.
- Words such as "slightly", "gently", "strongly", or "sharply" are allowed only when the time phase is clear and the wording does not replace important numeric evidence.
- The final caption does not need to mention every unit number, but it must not collapse distinct motion phases into one vague movement.
- For long clips, group adjacent units with the same dominant motion into longer phases instead of listing every unit mechanically.

Rules:
- Describe only camera motion.
- Do not mention scene objects, room type, furniture, doors, windows, people, or other visual content.
- Do not invent a target object.
- Do not mention pose matrices, raw metadata, implementation details, dataset internals, or field names in the output.
- Do not overstate weak motion.
- Do not ignore meaningful direction changes.
- The final caption should be useful for a VLM that will later caption the actual video frames.

Required JSON schema:
{
  "motion_units": [
    {
      "unit_index": <int>,
      "time_range_s": [<float>, <float>],
      "numeric_evidence": {
        "dominant_translation": "forward|backward|left|right|up|down|none|mixed",
        "dominant_rotation": "pan_left|pan_right|tilt_up|tilt_down|roll_left|roll_right|none|mixed",
        "translation_magnitude": "none|slight|moderate|large",
        "rotation_magnitude": "none|slight|moderate|large",
        "is_curved_motion": <bool>,
        "is_nearly_static": <bool>
      },
      "motion_terms": ["string", "..."],
      "description": "one concise sentence with the time range and only the meaningful rounded quantities"
    }
  ],
  "overall_motion_terms": ["string", "..."],
  "motion_caption": "one concise paragraph describing the full camera motion in temporal order, using meaningful rounded quantities only when they help"
}

Camera motion statistics:
<MOTION_EXTRACTION_JSON>
"""


VIDEO_CAPTIONING_TEMPLATE = """You are generating training metadata for a scene-grounded video generation model.

You receive:
A. a <CLIP_SECONDS>-second video clip of a static indoor scene,
B. a paired camera-motion caption derived from numeric camera poses of the clip,
C. an approximate timeline for the frames sampled by the video-language model.

Return JSON only. Do not include Markdown.

Use the video as visual evidence and the motion caption as guidance. Prioritize camera motion, visible scene layout, and object grounding over generic visual captioning.

Source usage:
- Use the video frames as the source of truth for visible objects, room layout, and visual details.
- Use the camera-motion caption as the source of truth for the camera-motion direction, magnitude, and temporal order.
- Use the sampled-frame timeline to identify where each visible frame sits in the entire clip. Refer to motion phases by approximate seconds when the evidence supports it.
- Ground the camera motion to visible objects or regions when the video clearly supports it.
- Do not describe objects or regions that are not visible.

Describe:
1. the initial viewpoint,
2. the camera motion over time, with approximate timestamps or phases,
3. the major static objects and room layout,
4. objects or regions the camera approaches, leaves, reveals, passes by, or turns toward.

Rules:
- Do not invent objects.
- Do not describe people unless they are clearly visible.
- Do not describe object motion unless it is clearly visible.
- Treat the scene as static unless the video clearly shows otherwise.
- Do not introduce lighting changes, time-of-day changes, weather, or environment changes.
- You may use approximate natural quantities such as "about 45 degrees", "about 90 degrees", or "roughly one meter" only when supported by the motion caption.
- Tie visible changes to time when possible, such as "at the start", "around 2 seconds", "during the third second", or equivalent phase wording when describing visible changes.
- Preserve rounded motion quantities from the camera-motion caption when they clarify camera control.
- Do not mention hidden metadata, camera poses, trajectory statistics, implementation details, or dataset internals.
- Focus on what a user would need to prompt a video generator.

Required JSON schema:
{
  "initial_view": "string",
  "temporal_motion_grounding": [
    {
      "time_range_s": [<float>, <float>],
      "visible_evidence": "what changes visually in this interval",
      "camera_motion": "motion in this interval"
    }
  ],
  "scene_objects": ["string", "..."],
  "revealed_or_emphasized_objects": ["string", "..."],
  "video_caption": "single detailed paragraph with approximate temporal grounding"
}

Camera-motion caption:
<MOTION_CAPTION_JSON>

Sampled frame timeline:
<VIDEO_FRAME_TIMELINE_JSON>
"""


IMAGE_CAPTIONING_TEMPLATE = """You are captioning one reference image for a scene-grounded video generation dataset.

Return JSON only. Do not include Markdown.

Use this image as visual evidence only. Describe what is visible in the image, especially details that are useful for cross-referencing this reference image from a later video-generation prompt.

Describe:
1. the room type, if apparent,
2. major objects and furniture,
3. spatial layout,
4. distinctive landmarks useful for cross-reference,
5. visible doorways, windows, walls, floor, and ceiling,
6. what viewpoint this image seems to show.

Rules:
- Do not invent objects.
- Be specific about object identity and position.
- Avoid vague phrases like "some furniture" when objects are identifiable.
- If the room type is unclear, use "unknown" rather than guessing.
- Do not describe camera motion; this is a single still reference image.
- Do not mention hidden metadata, file names, implementation details, or dataset internals.

Required JSON schema:
{
  "room_type": "string_or_unknown",
  "viewpoint": "string",
  "major_objects": ["string", "..."],
  "spatial_layout": "string",
  "distinctive_anchors": ["string", "..."],
  "image_caption": "single detailed paragraph"
}
"""


CAPTION_WIRING_TEMPLATE = """You are wiring captions for a scene-grounded video generation dataset.

You receive:
A. a detailed caption of the video,
B. structured captions of all reference images, including their reference indices and whether each one is the video start frame.

Return JSON only. Do not include Markdown.

Rewrite the video caption so it is wired up to the reference images in a way that a user could naturally prompt a video generator.

Source hierarchy:
- The video caption is the source of truth for the generated video's visual content, camera motion, and temporal order.
- The reference image captions are the only source of truth for what appears in each reference image.
- A reference-image claim is valid only if that specific reference image caption supports it.
- Wiring is compulsory when support is clear: if a visible object, area, target region, layout relation, layout anchor, or start view from the video caption clearly appears in one or more reference image captions, explicitly cross-refer to at least one supported reference image.
- Start-frame designation is compulsory: if any reference image is marked as the video start frame, the wired caption must state exactly which reference image is the starting viewpoint.

Rules:
- Preserve all important video motion.
- Preserve quantitative motion scale and temporal phase boundaries when the video caption or motion-guided caption supports them.
- Preserve objects visible in the video even if they are absent from the reference images.
- If a visible object or area appears in multiple reference images, any supported single reference or supported set of references is acceptable, e.g. "the first image", "the second image", or "the first and second images".
- Prefer wiring to all clearly useful reference images for start views, target objects, revealed regions, and distinctive landmarks, even if the caption becomes verbose.
- If the first frame is included as a reference, explicitly state which reference image is the starting viewpoint, e.g. "starting from the second image".
- Do not leave the start-frame fact only in "start_frame_reference"; include it directly in "wired_caption".
- If the first frame is not present, describe the starting viewpoint visually instead.
- Do not claim an object appears in a reference image unless the reference caption supports it.
- Do not force unrelated reference images into the caption. Use every reference that clearly helps identify the start view, target objects, emphasized regions, or spatial layout.
- Do not introduce environment changes such as nighttime, lights on/off, weather, or new objects unless the video caption supports them.
- Do not mention "caption", "metadata", "dataset", "camera poses", or "motion caption".
- Keep this as a descriptive intermediate caption, not yet an imperative user prompt.
- Preserve a natural reading order: start view first, then camera motion, then revealed or emphasized regions.
- Verbosity is acceptable.

Required JSON schema:
{
  "start_frame_reference": {
    "is_present": <bool>,
    "reference_index": <int_or_null>,
    "phrase": "string"
  },
  "reference_links": [
    {
      "reference_indices": [<int>, ...],
      "mentioned_as": "first image|second image|...",
      "objects_or_regions_used": ["string", "..."]
    }
  ],
  "objects_visible_in_video_but_not_references": ["string", "..."],
  "wired_caption": "string"
}

Video caption:
<VIDEO_CAPTION_JSON>

Reference image captions:
<IMAGE_CAPTIONS_JSON>
"""


CAPTION_REPHRASING_TEMPLATE = """You are rephrasing a wired video caption of a static scene into an instruction-style user prompt for a scene-grounded video generation model.

Return JSON only. Do not include Markdown.

The prompt should sound like a real user instructing a model to generate a video from reference images.

Rules:
- Use imperative phrasing.
- Mention the starting viewpoint clearly.
- Describe camera movement over <CLIP_SECONDS>.
- Preserve supported image cross-references from the wired caption when they identify the start view, target objects, emphasized regions, or spatial layout.
- Preserve the exact start-frame reference when the wired caption identifies one, such as "start from the second image".
- Preserve meaningful rounded distances, angles, and phase timing from the wired caption.
- Preserve scene geometry and object layout.
- Preserve objects or regions that the camera approaches, leaves, reveals, passes by, or turns toward.
- Do not mention hidden metadata, numeric poses, captions, datasets, manifests, or internal fields.
- Do not add visual content not supported by the wired caption.
- Do not introduce lighting changes, time-of-day changes, weather, scene transitions, or new objects.
- Do not overfit to dataset language.
- Avoid overly technical phrasing; make it sound like a natural user request.
- Keep the prompt detailed enough to guide generation, but not cluttered with redundant wording.

Required JSON schema:
{
  "synthesized_prompt": "string"
}

Wired caption:
<WIRED_CAPTION_JSON>
"""


CRITIC_JUDGING_TEMPLATE = """You are judging a synthesized user prompt for a scene-grounded video generation dataset.

You receive:
A. a natural-language camera-motion caption,
B. a detailed caption of the ground-truth video,
C. structured captions of all reference images,
D. the synthesized prompt.

Return JSON only. Do not include Markdown.

The prompt was synthesized from multiple sources using large models. Check whether it is faithful, useful, properly grounded, and suitable as an instruction to a video generation model.

Source hierarchy:
- The camera-motion caption is the main source of truth for camera motion.
- The video caption is the main source of truth for visible scene content, layout, and what the camera approaches, leaves, reveals, or turns toward.
- The reference image captions are the only source of truth for claims about what appears in each reference image. Additionally, they are ordered.
- The synthesized prompt is the candidate output to validate.
- A claim is supported only if it is clearly implied by one or more of the provided sources.

Important distinction:
- It is acceptable for the synthesized prompt to mention objects visible in the video even if they are not visible in the reference images.
- It is not acceptable to claim that an object appears in a specific reference image unless that reference image caption supports it.
- It is acceptable to paraphrase motion or visual content.
- It is not acceptable to reverse motion direction, invent objects, invent environment changes, or introduce hidden metadata.

Validation rules:
- Do not judge whether the prompt is stylish or beautiful.
- Only check whether each rule is satisfied.
- For each rule, output true only when the inputs clearly support it.
- If uncertain, output false for that rule.
- Do not output an overall score. The program will compute the score.
- Be conservative: false rejection is better than accepting a bad prompt.

Fatal checks:
- "valid_json_inputs_understood": true only if the provided inputs are understandable and sufficient for validation.
- "reference_captions_have_correct_indices": true only if each reference-image caption correctly refers to its own image number or ordinal, which is the order it appears in the captions text.
- "reference_captions_have_no_conflicting_start_frame_claims": true only if the image captions do not make conflicting claims about which reference image is the video start frame.
- "prompt_is_non_empty_instruction": true only if the synthesized prompt is non-empty and written as a user instruction.
- "no_hallucinated_new_objects": true only if the prompt does not introduce objects, furniture, people, animals, text, screens, decorations, or regions unsupported by the video caption or reference captions.
- "no_hallucinated_environment_change": true only if the prompt does not invent lighting changes, time-of-day changes, weather, scene transitions, object movement, room transformations, or new objects appearing/disappearing.
- "no_unsupported_dynamic_content": true only if the prompt does not invent moving people, moving objects, animated actions, or non-static scene behavior unsupported by the video caption.
- "no_invalid_reference_index": true only if every referenced image index exists in the reference image captions.
- "no_unverified_reference_claim": true only if every claim about a specific reference image is supported by that reference image caption.
- "no_reference_claim_without_specific_index": true only if claims about reference images name the specific image index or indices instead of vaguely saying "the reference image" when multiple references exist.
- "no_missing_start_frame_reference": true only if a start-frame reference is available and the prompt uses the correct image as the starting viewpoint, or if no start-frame reference is available and the prompt describes the starting viewpoint visually.
- "no_start_frame_reference_error": true only if the prompt does not incorrectly identify which reference image is the video start frame.
- "no_contradictory_camera_motion": true only if the prompt does not contradict the camera-motion caption or video caption in direction, order, scale, or type of motion.
- "no_opposite_motion_direction": true only if the prompt does not reverse any important motion direction, such as left versus right, forward versus backward, up versus down, pan left versus pan right, or tilt up versus tilt down.
- "no_invalid_zoom_claim": true only if the prompt does not describe pose-derived forward/backward camera motion as optical zoom unless the video caption explicitly supports zoom.
- "no_temporal_or_duration_contradiction": true only if the prompt does not contradict the duration or the temporal order of the motion.
- "no_scene_identity_contradiction": true only if the prompt does not change the type, identity, or layout of the scene in a way unsupported by the video or reference captions.
- "no_hidden_metadata_leak": true only if the prompt does not mention hidden metadata, implementation details, dataset internals, pose matrices, numeric camera poses, trajectory statistics, JSON, captions, manifests, ScanNet++, or file paths.

Quality checks:
- "has_clear_start_viewpoint": true only if the prompt clearly establishes the initial view of the generated video.
- "uses_start_frame_reference_if_available": true if no first-frame reference is available, or if it is available and the prompt correctly uses it as the starting viewpoint.
- "describes_start_view_visually_if_no_start_reference": true if a first-frame reference is available, or if none is available and the prompt instead describes the starting viewpoint visually.
- "has_meaningful_camera_motion": true only if the prompt contains a clear camera movement rather than only describing a static scene.
- "aligns_with_motion_caption": true only if the prompt's camera motion aligns with the camera motion caption.
- "has_temporally_grounded_motion": true only if the prompt preserves major motion phases in temporal order, using approximate seconds or clear beginning/middle/end grounding when supported.
- "has_quantitative_motion_scale_when_supported": true only if the prompt preserves approximate distances, angles, or relative motion scale from the motion/video captions when those quantities are available.
- "aligns_with_video_caption": true only if the prompt's content aligns with the video caption.
- "preserves_motion_order": true only if the prompt preserves the temporal order of major motion phases.
- "preserves_motion_phase_boundaries": true only if distinct motion phases in the motion/video captions are not collapsed into an ambiguous single movement.
- "preserves_motion_strength": true only if the prompt does not exaggerate weak motion or understate dominant motion in a misleading way.
- "does_not_replace_camera_translation_with_zoom": true only if forward/backward camera movement remains physical camera motion unless optical zoom is explicitly supported.
- "grounds_camera_motion_to_visible_regions": true only if camera moves such as approaching, passing, revealing, or turning are grounded to visible scene regions when the video caption supports such grounding.
- "mentions_relevant_scene_objects": true only if the prompt mentions important scene objects or regions needed to ground the camera motion.
- "objects_are_supported_by_video_caption": true only if objects mentioned as part of the generated video are supported by the video caption.
- "uses_supported_reference_images_compulsorily": true only if the prompt cross-refers to reference images for visible objects, start views, target regions, or layout anchors when the reference captions clearly support those links.
- "reference_image_mentions_are_supported": true only if every reference image mention is supported by the corresponding reference image caption.
- "preserves_start_reference_grounding": true only if the prompt preserves the start-frame reference or visual start-view description from the inputs.
- "preserves_target_reference_grounding": true only if objects or regions approached, revealed, passed, or turned toward are wired to supporting reference images when such images exist.
- "uses_reference_images_when_helpful": true only if the prompt uses reference images naturally when they help identify the start view, target object, emphasized region, or layout.
- "uses_multiple_references_when_clearly_helpful": true only if the prompt uses multiple reference images when that materially improves grounding of the start view, target object, or layout.
- "does_not_overuse_reference_images": true only if the prompt avoids unnecessary or repetitive reference-image mentions.
- "preserves_video_only_visible_objects_when_needed": true only if important objects visible in the video but absent from the references are still preserved when they matter for the motion or scene grounding.
- "captures_revealed_or_emphasized_regions": true only if the prompt preserves regions or objects that the camera approaches, leaves, reveals, passes by, or turns toward.
- "preserves_spatial_layout": true only if the prompt keeps the important room layout and object relationships.
- "preserves_static_scene_assumption": true only if the prompt treats the scene as static unless the video caption clearly supports otherwise.
- "is_generation_instruction": true only if the prompt is phrased as an instruction to generate a video.
- "uses_natural_user_language": true only if the prompt sounds like a natural user request rather than a data annotation.
- "is_not_plain_caption": true only if the prompt is not merely a descriptive caption.
- "is_not_too_generic": true only if the prompt contains enough scene and motion detail to be useful for generation.
- "is_not_overly_verbose_or_redundant": true only if the prompt is not cluttered with unnecessary repetition.
- "does_not_sound_like_dataset_metadata": true only if the prompt avoids terms such as "metadata", "caption", "ground truth", "sample", "dataset", or similar annotation language.
- "does_not_mention_pose_or_trajectory_metadata": true only if the prompt avoids terms such as "pose", "trajectory statistics", "camera-to-world", "matrix", "translation_right_m", or similar internal fields.

Required JSON schema:
{
  "fatal_checks": {
    "valid_json_inputs_understood": <bool>,
		"reference_captions_have_correct_indices": <bool>,
    "reference_captions_have_no_conflicting_start_frame_claims": <bool>,
    "prompt_is_non_empty_instruction": <bool>,
    "no_hallucinated_new_objects": <bool>,
    "no_hallucinated_environment_change": <bool>,
    "no_unsupported_dynamic_content": <bool>,
    "no_invalid_reference_index": <bool>,
    "no_unverified_reference_claim": <bool>,
    "no_reference_claim_without_specific_index": <bool>,
    "no_missing_start_frame_reference": <bool>,
    "no_start_frame_reference_error": <bool>,
    "no_contradictory_camera_motion": <bool>,
    "no_opposite_motion_direction": <bool>,
    "no_invalid_zoom_claim": <bool>,
    "no_temporal_or_duration_contradiction": <bool>,
    "no_scene_identity_contradiction": <bool>,
    "no_hidden_metadata_leak": <bool>
  },
  "quality_checks": {
    "has_clear_start_viewpoint": <bool>,
    "uses_start_frame_reference_if_available": <bool>,
    "describes_start_view_visually_if_no_start_reference": <bool>,
    "has_meaningful_camera_motion": <bool>,
    "aligns_with_motion_caption": <bool>,
    "has_temporally_grounded_motion": <bool>,
    "has_quantitative_motion_scale_when_supported": <bool>,
    "aligns_with_video_caption": <bool>,
    "preserves_motion_order": <bool>,
    "preserves_motion_phase_boundaries": <bool>,
    "preserves_motion_strength": <bool>,
    "does_not_replace_camera_translation_with_zoom": <bool>,
    "grounds_camera_motion_to_visible_regions": <bool>,
    "mentions_relevant_scene_objects": <bool>,
    "objects_are_supported_by_video_caption": <bool>,
    "uses_supported_reference_images_compulsorily": <bool>,
    "reference_image_mentions_are_supported": <bool>,
    "preserves_start_reference_grounding": <bool>,
    "preserves_target_reference_grounding": <bool>,
    "uses_multiple_references_when_clearly_helpful": <bool>,
    "does_not_overuse_reference_images": <bool>,
    "preserves_video_only_visible_objects_when_needed": <bool>,
    "captures_revealed_or_emphasized_regions": <bool>,
    "preserves_spatial_layout": <bool>,
    "preserves_static_scene_assumption": <bool>,
    "is_generation_instruction": <bool>,
    "uses_natural_user_language": <bool>,
    "is_not_plain_caption": <bool>,
    "is_not_too_generic": <bool>,
    "is_not_overly_verbose_or_redundant": <bool>,
    "does_not_sound_like_dataset_metadata": <bool>,
    "does_not_mention_pose_or_trajectory_metadata": <bool>
  }
}

Camera-motion caption:
<MOTION_CAPTION_JSON>

Video caption:
<VIDEO_CAPTION_JSON>

Reference image captions:
<IMAGE_CAPTIONS_JSON>

Synthesized prompt:
<SYNTHESIZED_PROMPT>
"""


DISTILLATION_TEMPLATE = """You are creating shorter variants of a prompt for a video generation dataset.

Return JSON only. Do not include Markdown.

Create:
1. medium prompt: shorter than the original prompt, but still preserving the start viewpoint, dominant camera motion, and main target objects or regions.
2. coarse prompt: a natural user-level prompt that is short and underspecified, optionally omitting secondary motion details, secondary visual details, and nonessential reference-image links while staying consistent with the original prompt.

Rules:
- Medium and coarse prompts should be consistent compressions, not new interpretations.
- Do not add new objects, environment changes, or camera motion.
- Do not contradict the original prompt.
- Preserve the starting viewpoint when the original prompt establishes one.
- Preserve the exact start-frame reference image when the original prompt uses one; this is a crucial fact, not a trivial detail.
- Preserve dominant camera direction, dominant scale, and the main target objects or regions. Omit only tiny motion components, secondary visual details, or redundant wording.
- The medium prompt should keep the dominant camera motion and the main visual grounding, but may omit fine-grained timing, small motion components, or secondary references.
- The coarse prompt may omit secondary reference-image details, exact minor motion quantities, minor objects, and temporal specifics when they are not essential to stay consistent.
- Keep image cross-references when they are essential for understanding the start view or target object.
- The coarse prompt should sound like a normal short user request, not a dataset annotation or a compressed checklist.
- Do not mention hidden metadata, captions, datasets, manifests, or internal fields.

Required JSON schema:
{
  "medium_prompt": "string",
  "coarse_prompt": "string"
}

Original prompt:
<SYNTHESIZED_PROMPT>
"""
