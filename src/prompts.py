MOTION_DIGESTING_TEMPLATE = """You are digesting numeric camera-motion extraction into a camera-motion caption.

The whole clip is split into motion units. You receive statistics for the overall motion and all motion units. Fields with the "local" prefix in each motion unit are computed relative to the first pose of that unit; they describe the local motion within that unit. Fields with the "first_pose" prefix in each motion unit are computed relative to the first pose of the whole clip; they describe where the unit begins within the full motion.

Return JSON only. Do not include Markdown.

Camera-coordinate convention:
- The poses are camera-to-world matrices.
- Relative motion is expressed in the local camera coordinate system of the starting pose.
- +X means camera-right.
- +Y means camera-down.
- +Z means camera-forward.
- All translations and rotations describe camera motion, not object motion.

Input field meanings:
- "duration_s": duration of the whole video clip.
- "trajectory_length_m": accumulated path length of the camera over the whole clip. This is always non-negative and can be larger than the straight-line translation distance.
- "translation_right_m": net rightward displacement of the camera from the first pose to the last pose. Positive means moving right; negative means moving left.
- "translation_down_m": net downward displacement of the camera from the first pose to the last pose. Positive means moving down or lowering; negative means moving up or rising.
- "translation_forward_m": net forward displacement of the camera from the first pose to the last pose. Positive means moving forward, pushing in, or approaching; negative means moving backward, pulling back, or retreating.
- "translation_distance_m": straight-line distance between the first and last camera positions. This is always non-negative.
- "delta_yaw_deg": net yaw change from the first pose to the last pose. Positive means turning/panning right; negative means turning/panning left.
- "delta_pitch_deg": net pitch change from the first pose to the last pose. Positive means tilting up; negative means tilting down.
- "delta_roll_deg": net roll change from the first pose to the last pose. Positive means rolling clockwise; negative means rolling counterclockwise.
- "motion_units": fixed-duration temporal units of the whole clip.
- "time_range_s": start and end time of the motion unit within the clip.
- "first_pose_position_m": where the motion unit begins, expressed relative to the first pose of the whole clip.
- "first_pose_yaw_deg", "first_pose_pitch_deg", and "first_pose_roll_deg": the camera orientation at the beginning of the motion unit, expressed relative to the first pose of the whole clip. These describe the starting state of the unit, not the motion within the unit.
- "local_trajectory_length_m": accumulated path length within the motion unit.
- "local_translation_right_m", "local_translation_down_m", and "local_translation_forward_m": net displacement within the motion unit, expressed relative to the first pose of that unit.
- "local_translation_distance_m": straight-line distance between the first and last camera positions within the motion unit.
- "local_delta_yaw_deg", "local_delta_pitch_deg", and "local_delta_roll_deg": orientation change within the motion unit.

Your task:
- Interpret each motion unit as a camera motion phrase.
- Use the numeric values to choose camera-motion terms.
- Produce compact intermediate motion evidence for each motion unit.
- Write a short description for each motion unit.
- Write one final detailed camera-motion caption that encapsulates all motion.

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
- pass by, move past, move through, enter, exit, reveal, settle toward,
- pause, hold, settle, remain still, stay steady, static, nearly still,
- slowly, smoothly, gradually, gently, steadily, quickly, abruptly, slightly, strongly, sharply.

Numeric interpretation hints:
- Right translation: positive means sliding/trucking/tracking right; negative means sliding/trucking/tracking left.
- Down translation: positive means lowering or moving down; negative means rising or moving up.
- Forward translation: positive means moving forward, pushing in, or approaching; negative means moving backward, pulling back, or retreating.
- Yaw delta: positive means panning/turning right; negative means panning/turning left.
- Pitch delta: positive means tilting up; negative means tilting down.
- Roll delta: positive means rolling clockwise; negative means rolling counterclockwise.
- Use the local fields to describe motion within each motion unit.
- Use the first_pose fields only to understand how the motion units connect over time.
- If trajectory_length_m is much larger than translation_distance_m, the motion may be curved, indirect, or back-and-forth.
- Meaningful translation plus meaningful yaw can often be described as an arc, curve, sweep, or veer.
- Forward motion plus lateral motion may be described as a diagonal move or a curved approach.
- Lateral motion plus yaw in the same general direction may be described as a smooth arc or sweeping move.
- Lateral motion plus yaw in the opposite direction may indicate tracking while looking back or panning across the scene.
- Very small translation and very small rotation should be described as mostly static or nearly still.
- Pose-derived forward motion should not be described as zoom.
- Use "orbit" only when the numeric trajectory strongly suggests curved motion around a center.
- Prefer simple motion descriptions when the evidence is ambiguous.
- Be conservative with strong terms such as "sharp", "quickly", "orbit", or "large turn" unless the numbers clearly support them.
- The final caption should preserve temporal order: describe what happens first, then what happens next, then how the motion ends.

Rules:
- Describe only camera motion.
- Do not mention scene objects, room type, furniture, doors, windows, people, or any other visual content.
- Do not invent a target object.
- Do not mention pose matrices, raw metadata, implementation details, or dataset internals.
- Use approximate rounded quantities when helpful, such as "about 0.5 meters" or "about 60 degrees".
- Be conservative. If a motion is weak, describe it as slight or omit it.
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
        "is_curved_motion": <bool>
      },
      "motion_terms": ["string", "..."],
      "description": "one concise sentence describing this motion unit"
    }
  ],
  "overall_motion_terms": ["string", "..."],
  "motion_caption": "one concise paragraph describing the full camera motion"
}

Camera motion extraction:
<MOTION_EXTRACTION_JSON>
"""


VIDEO_CAPTIONING_TEMPLATE = """You are generating training metadata for a scene-grounded video generation model.

You receive:
A. a 5-second video clip of a static indoor scene,
B. a camera-motion caption derived from numeric camera poses.

Return JSON only. Do not include Markdown.

Use the video as visual evidence and the motion caption as guidance. Prioritize camera motion, visible scene layout, and object grounding over generic visual captioning.

Source usage:
- Use the video frames as the source of truth for visible objects, room layout, and visual details.
- Use the camera-motion caption as the source of truth for the camera-motion direction, magnitude, and temporal order.
- Ground the camera motion to visible objects or regions when the video clearly supports it.
- Do not describe objects or regions that are not visible.

Describe:
1. the initial viewpoint,
2. the camera motion over time,
3. the major static objects and room layout,
4. objects or regions the camera approaches, leaves, reveals, passes by, or turns toward.

Rules:
- Do not invent objects.
- Do not describe people unless they are clearly visible.
- Do not describe object motion unless it is clearly visible.
- Treat the scene as static unless the video clearly shows otherwise.
- Do not introduce lighting changes, time-of-day changes, weather, or environment changes.
- You may use approximate natural quantities such as "about 45 degrees", "about 90 degrees", or "roughly one meter" only when supported by the motion caption.
- Do not mention hidden metadata, camera poses, trajectory statistics, implementation details, or dataset internals.
- Focus on what a user would need to prompt a video generator.

Required JSON schema:
{
  "initial_view": "string",
  "scene_objects": ["string", "..."],
  "revealed_or_emphasized_objects": ["string", "..."],
  "video_caption": "single detailed paragraph"
}

Camera-motion caption:
<MOTION_CAPTION_JSON>
"""


IMAGE_CAPTIONING_TEMPLATE = """You are captioning one reference image for a scene-grounded video generation dataset.

The index of this image is <REF_INDEX>, and it <IS_VIDEO_START_CLAUSE> the start frame of the video. Explicitly mention this two facts in the caption.

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
- Always mention whether this image is the start frame of the video.
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
B. captions of all reference images.

Return JSON only. Do not include Markdown.

Rewrite the video caption so it is wired up to the reference images in a way that a user could naturally prompt a video generator.

Source hierarchy:
- The video caption is the source of truth for the generated video's visual content, camera motion, and temporal order.
- The reference image captions are the only source of truth for what appears in each reference image.
- A reference-image claim is valid only if that specific reference image caption supports it.

Rules:
- Preserve all important video motion.
- Preserve objects visible in the video even if they are absent from the reference images.
- If a visible object or area appears in a reference image, refer to that image naturally, e.g. "the desk visible in the first image".
- If the first frame is included as a reference, explicitly state which reference image is the starting viewpoint, e.g. "starting from the second image".
- If the first frame is not present, describe the starting viewpoint visually instead.
- Do not claim an object appears in a reference image unless the reference caption supports it.
- Do not force every reference image into the caption. Use only references that help identify the start view, target objects, or spatial layout.
- Do not introduce environment changes such as nighttime, lights on/off, weather, or new objects unless the video caption supports them.
- Do not mention "caption", "metadata", "dataset", "camera poses", or "motion caption".
- Keep this as a descriptive intermediate caption, not yet an imperative user prompt.
- Preserve a natural reading order: start view first, then camera motion, then revealed or emphasized regions.

Required JSON schema:
{
  "start_frame_reference": {
    "is_present": <bool>,
    "reference_index": <int_or_null>,
    "phrase": "string"
  },
  "reference_links": [
    {
      "reference_index": <int>,
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
- Cross-reference images naturally where useful.
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
- The reference image captions are the only source of truth for claims about what appears in each reference image.
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
- "prompt_is_non_empty_instruction": true only if the synthesized prompt is non-empty and written as a user instruction.
- "no_hallucinated_new_objects": true only if the prompt does not introduce objects, furniture, people, animals, text, screens, decorations, or regions unsupported by the video caption or reference captions.
- "no_hallucinated_environment_change": true only if the prompt does not invent lighting changes, time-of-day changes, weather, scene transitions, object movement, room transformations, or new objects appearing/disappearing.
- "no_unsupported_dynamic_content": true only if the prompt does not invent moving people, moving objects, animated actions, or non-static scene behavior unsupported by the video caption.
- "no_invalid_reference_index": true only if every referenced image index exists in the reference image captions.
- "no_unverified_reference_claim": true only if every claim about a specific reference image is supported by that reference image caption.
- "no_start_frame_reference_error": true only if the prompt does not incorrectly identify which reference image is the video start frame.
- "no_contradictory_camera_motion": true only if the prompt does not contradict the camera-motion caption or video caption in direction, order, scale, or type of motion.
- "no_opposite_motion_direction": true only if the prompt does not reverse any important motion direction, such as left versus right, forward versus backward, up versus down, pan left versus pan right, or tilt up versus tilt down.
- "no_invalid_zoom_claim": true only if the prompt does not describe pose-derived forward/backward camera motion as optical zoom unless the video caption explicitly supports zoom.
- "no_temporal_or_duration_contradiction": true only if the prompt does not contradict the 5-second duration or the temporal order of the motion.
- "no_scene_identity_contradiction": true only if the prompt does not change the type, identity, or layout of the scene in a way unsupported by the video or reference captions.
- "no_hidden_metadata_leak": true only if the prompt does not mention hidden metadata, implementation details, dataset internals, pose matrices, numeric camera poses, trajectory statistics, JSON, captions, manifests, ScanNet++, or file paths.

Quality checks:
- "has_clear_start_viewpoint": true only if the prompt clearly establishes the initial view of the generated video.
- "uses_start_frame_reference_if_available": true only if the first frame is included as a reference image and the prompt correctly uses it as the starting viewpoint.
- "describes_start_view_visually_if_no_start_reference": true only if the first frame is not included as a reference image and the prompt instead describes the starting viewpoint visually.
- "has_meaningful_camera_motion": true only if the prompt contains a clear camera movement rather than only describing a static scene.
- "aligns_with_motion_caption": true only if the prompt's camera motion aligns with the camera motion caption.
- "aligns_with_video_caption": true only if the prompt's content aligns with the video caption.
- "preserves_motion_order": true only if the prompt preserves the temporal order of major motion phases.
- "preserves_motion_strength": true only if the prompt does not exaggerate weak motion or understate dominant motion in a misleading way.
- "mentions_relevant_scene_objects": true only if the prompt mentions important scene objects or regions needed to ground the camera motion.
- "objects_are_supported_by_video_caption": true only if objects mentioned as part of the generated video are supported by the video caption.
- "reference_image_mentions_are_supported": true only if every reference image mention is supported by the corresponding reference image caption.
- "uses_reference_images_when_helpful": true only if the prompt uses reference images naturally when they help identify the start view, target object, or layout.
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
    "prompt_is_non_empty_instruction": <bool>,
    "no_hallucinated_new_objects": <bool>,
    "no_hallucinated_environment_change": <bool>,
    "no_unsupported_dynamic_content": <bool>,
    "no_invalid_reference_index": <bool>,
    "no_unverified_reference_claim": <bool>,
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
    "aligns_with_video_caption": <bool>,
    "preserves_motion_order": <bool>,
    "preserves_motion_strength": <bool>,
    "mentions_relevant_scene_objects": <bool>,
    "objects_are_supported_by_video_caption": <bool>,
    "reference_image_mentions_are_supported": <bool>,
    "uses_reference_images_when_helpful": <bool>,
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
1. medium prompt: shorter than the original prompt, but still preserving the start viewpoint, main camera motion, and main target objects or regions.
2. coarse prompt: a natural user-level prompt that is short and underspecified, but still consistent with the original prompt.

Rules:
- Medium and coarse prompts should be consistent compressions, not new interpretations.
- Do not add new objects, environment changes, or camera motion.
- Do not contradict the original prompt.
- Preserve the starting viewpoint if it is important.
- Preserve the dominant camera motion.
- Preserve the main target objects or regions that ground the motion.
- Keep image cross-references only if they are essential for understanding the start view or target object.
- The medium prompt may keep some reference-image details.
- The coarse prompt should sound like a normal short user request, not a dataset annotation.
- Do not mention hidden metadata, captions, datasets, manifests, or internal fields.

Required JSON schema:
{
  "medium_prompt": "string",
  "coarse_prompt": "string"
}

Original prompt:
<SYNTHESIZED_PROMPT>
"""
