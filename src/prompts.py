# TODO: Update this templates to match the implementation in synthesize.py
MOTION_DIGESTING_TEMPLATE = """You are converting numeric camera-motion extraction into a clear camera-motion caption.

The clip is split into motion units. You receive statistics for the overall motion and all motion units. Your goal is to Interpret all  describe the camera path in a way that a later video-captioning VLM can use to understand how the camera moves through time.

Return JSON only. Do not include Markdown.

Camera-coordinate convention:
- The poses are camera-to-world matrices.
- Relative motion is expressed in the local camera coordinate system of the starting pose.
- +X means camera-right.
- +Y means camera-down.
- +Z means camera-forward.
- All translations and rotations describe camera motion, not object motion.

Fields with the "local" prefix in each motion unit are computed relative to the first pose of that unit; they describe the local motion within that unit. Fields with the "first_pose" prefix in each motion unit are computed relative to the first pose of the whole clip; they describe where the unit begins within the full motion.

Field meanings and units:
- "duration_s": clip or unit duration, in seconds.
- "trajectory_length_m": accumulated camera path length, in meters. It is non-negative and can exceed the straight-line translation distance.
- "translation_right_m": net camera displacement along camera-right, in meters.
- "translation_down_m": net camera displacement along camera-down, in meters.
- "translation_forward_m": net camera displacement along camera-forward, in meters.
- "translation_distance_m": straight-line distance between first and last camera positions, in meters.
- "delta_yaw_deg": net yaw change, in degrees.
- "delta_pitch_deg": net pitch change, in degrees.
- "delta_roll_deg": net roll change, in degrees.
- "rotation_magnitude_deg": combined yaw, pitch, and roll magnitude, in degrees, if available.
- "net_rotation_angle_deg": shortest overall orientation change, in degrees, if available.
- "angular_path_deg": accumulated frame-to-frame rotation, in degrees, if available.
- "path_linearity": net translation distance divided by trajectory length, if available. Lower values suggest curved, indirect, or back-and-forth motion.
- "translation_jitter", "rotation_jitter", "direction_reversal_fraction": dimensionless shake or instability indicators, if available.
- "max_step_distance_m": largest frame-to-frame translation jump, in meters, if available.
- "max_rotation_step_deg": largest frame-to-frame rotation jump, in degrees, if available.
- "path_to_translation_distance_ratio": accumulated path length divided by net translation distance, if available.
- "rotation_path_to_net_ratio": accumulated rotation divided by net rotation, if available.
- "motion_units": fixed-duration temporal units of the whole clip.
- "time_range_s": start and end time of a motion unit within the clip, in seconds.
- "first_pose_position_m": where the unit begins relative to the first pose of the whole clip, as [right, down, forward] in meters.
- "first_pose_yaw_deg", "first_pose_pitch_deg", "first_pose_roll_deg": camera orientation at the beginning of the unit relative to the first pose of the whole clip, in degrees. These describe the starting state of the unit, not the motion inside the unit.
- Fields with the "local_" prefix describe only motion within that unit, using the same units and signs as the corresponding overall fields.

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
- Use numeric values only when they are meaningful for understanding the motion.
- Do not force every small number into the text.
- Prefer natural rounded quantities over raw precision.
- Avoid vague motion words when they hide important numeric evidence.

Motion interpretation guidance:
- Use local fields to describe motion within each unit.
- Use first_pose fields only to understand how units connect into a continuous path.
- If accumulated path is much larger than net displacement, interpret conservatively as curved, indirect, back-and-forth, or shaky according to the per-unit sequence.
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
- Round times to about 0.1 seconds.
- Round translations naturally: for small but meaningful motion, use about 0.05 m or 0.1 m precision; for larger motion, use about 0.1 m or coarser.
- Round rotations naturally: usually to the nearest 1, 5, 10, or 15 degrees depending on scale.
- Avoid raw-looking values such as 0.237 m or 13.842 degrees.
- Words such as "slightly", "gently", "strongly", or "sharply" are allowed only when the time phase is clear and the wording does not replace important numeric evidence.
- The final caption does not need to mention every unit number, but it must not collapse distinct motion phases into one vague movement.

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

Camera motion extraction:
<MOTION_EXTRACTION_JSON>
"""


VIDEO_CAPTIONING_TEMPLATE = """You are generating training metadata for a scene-grounded video generation model.

You receive:
A. a 5-second video clip of a static indoor scene,
B. a camera-motion caption derived from numeric camera poses,
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

The index of this image is <REF_INDEX>, and it <IS_VIDEO_START_CLAUSE> the start frame of the video. Explicitly mention these two facts in the caption.

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
- Refer to this image by its index, such as "the first image", "the third image", "image 6", "image 12", etc. But avoid using the word "index" directly.
- Do not refer to this image as a different ordinal or number.
- Always mention whether this image is the start frame of the video.
- If the room type is unclear, use "unknown" rather than guessing.
- Do not describe camera motion; this is a single still reference image.
- Do not mention hidden metadata, file names, implementation details, or dataset internals.

Required JSON schema:
{
  "reference_index": <int>,
  "is_video_start_frame": <bool>,
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
B. captions of all reference images.

Return JSON only. Do not include Markdown.

Rewrite the video caption so it is wired up to the reference images in a way that a user could naturally prompt a video generator.

Source hierarchy:
- The video caption is the source of truth for the generated video's visual content, camera motion, and temporal order.
- The reference image captions are the only source of truth for what appears in each reference image.
- A reference-image claim is valid only if that specific reference image caption supports it.
- Wiring is compulsory when support is clear: if a visible object, area, target region, layout relation, layout anchor, or start view from the video caption clearly appears in one or more reference image captions, explicitly cross-refer to at least one supported reference image.

Rules:
- Preserve all important video motion.
- Preserve objects visible in the video even if they are absent from the reference images.
- If a visible object or area appears in multiple reference images, any supported single reference or supported set of references is acceptable, e.g. "the first image", "the second image", or "the first and second images".
- Prefer wiring to all clearly useful reference images for start views, target objects, revealed regions, and distinctive landmarks, even if the caption becomes verbose.
- If the first frame is included as a reference, explicitly state which reference image is the starting viewpoint, e.g. "starting from the second image".
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


CAPTION_REPHRASING_TEMPLATE = """You are rephrasing a wired scene/video caption into an instruction-style user prompt for a scene-grounded video generation model.

Return JSON only. Do not include Markdown.

The prompt should sound like a real user instructing a model to generate a video from reference images.

Rules:
- Use imperative phrasing.
- Mention the starting viewpoint clearly.
- Describe camera movement over 5 seconds.
- Preserve supported image cross-references from the wired caption when they identify the start view, target objects, emphasized regions, or spatial layout.
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
C. captions of all reference images,
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
- "no_temporal_or_duration_contradiction": true only if the prompt does not contradict the 5-second duration or the temporal order of the motion.
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
- "aligns_with_motion_caption": true only if the prompt's camera motion aligns with the camera motion caption.
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
    "aligns_with_motion_caption": <bool>,
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
    "does_not_overuse_reference_images": <bool>,
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
- Preserve the starting viewpoint if it is important.
- The medium prompt should keep the dominant camera motion and the main visual grounding, but may omit fine-grained timing, small motion components, or secondary references.
- The coarse prompt may omit reference-image details, exact motion quantities, minor objects, and temporal specifics when they are not essential to stay consistent.
- Keep image cross-references only if they are essential for understanding the start view or target object.
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
