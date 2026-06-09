MOTION_DIGESTING_TEMPLATE = """You are converting numeric camera-motion statistics of a <CLIP_SECONDS>-second video clip into a clear camera-motion caption.

The clip is split into chronological motion units of <UNIT_SECONDS> seconds. Use the overall statistics for the dominant full-clip motion, and use motion units to preserve phase order, direction changes, brief holds, and changes in motion strength.

Return JSON only. Do not include Markdown.

Camera-coordinate convention:
- Poses are camera-to-world matrices.
- Relative motion is expressed in the local camera coordinate system of the starting pose.
- +X means camera-right, +Y means camera-down, and +Z means camera-forward.
- All translations and rotations describe camera motion, not object motion.
- Do not mention scene objects, room type, furniture, doors, windows, people, or other visual content.

Field interpretation:
- "overall_motion" summarizes the full clip from first sampled pose to last sampled pose.
- "motion_units" is a chronological list of fixed-duration temporal segments.
- "unit_start_in_clip_start_*" fields describe where a unit begins in the whole clip. Use them only to connect phases.
- "unit_end_in_unit_start_*" fields describe local motion inside that unit. Use them as the main evidence for each phase.
- Times are seconds, translations are meters, and rotations are degrees.

Signed-value interpretation:
- Right translation: positive means slide/truck/track right; negative means slide/truck/track left.
- Down translation: positive means lower or move down; negative means rise or move up.
- Forward translation: positive means move forward, push in, or approach; negative means move backward, pull back, or retreat.
- Yaw delta: positive means pan or turn right; negative means pan or turn left.
- Pitch delta: positive means tilt up; negative means tilt down.
- Roll delta: positive means roll clockwise; negative means roll counterclockwise.

Writing policy:
- Preserve temporal order and meaningful phase boundaries.
- Quantify dominant motion with natural rounded values. Include meters/degrees for meaningful whole-clip motion and for unit phases above about 0.10 m or 5 degrees.
- Omit tiny components unless they mark a real direction change. Treat units below about 0.05 m and 2 degrees as nearly still.
- If path length is much larger than chord length, describe the motion as curved, indirect, wavering, or back-and-forth according to the unit sequence.
- Translation plus yaw may be an arc, curve, sweep, or veer when supported. Use "orbit" only when the trajectory strongly supports movement around a center.
- Pose-derived forward/backward motion is physical camera movement, not optical zoom.
- Round times to about 0.1 seconds. Round translations naturally to about 0.05 m, 0.1 m, or coarser. Round rotations naturally to about 1, 5, 10, or 15 degrees depending on scale.
- Avoid raw-looking values such as 0.237 m or 13.842 degrees.
- For long clips, group adjacent units with the same dominant motion instead of listing every unit mechanically.

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
      "description": "one concise sentence with the time range and only meaningful rounded quantities"
    }
  ],
  "overall_motion_terms": ["string", "..."],
  "motion_caption": "one concise paragraph describing the full camera motion in temporal order with meaningful rounded quantities"
}

Camera motion statistics:
<MOTION_EXTRACTION_JSON>
"""


VIDEO_CAPTIONING_TEMPLATE = """You are generating training metadata for a scene-grounded video generation model.

You receive:
A. a <CLIP_SECONDS>-second video clip of a static indoor scene,
B. a camera-motion caption derived from numeric camera poses of the clip,
C. an approximate timeline for the frames sampled by the video-language model.

Return JSON only. Do not include Markdown.

Use the video as visual evidence and the motion caption as guidance. Prioritize fine-grained camera-motion control over generic visual captioning; reference images will carry most scene appearance later.

Source usage:
- Use the video frames as the source of truth for visible objects, room layout, and visual details.
- Use the camera-motion caption as the source of truth for camera-motion direction, magnitude, rounded quantities, and temporal order.
- Use the sampled-frame timeline to identify where each visible frame sits in the entire clip.
- Ground the camera motion to visible objects or regions when the video clearly supports it.
- Do not describe objects or regions that are not visible.
- Do not drop camera-motion quantities from the motion caption. Carry meters/degrees forward for the final user prompt.

Describe:
1. the initial viewpoint,
2. the camera motion over time, with approximate timestamps or phases and concrete rounded amounts from the motion caption,
3. only the major static objects and layout anchors needed to ground the camera path,
4. objects or regions the camera approaches, leaves, reveals, passes by, or turns toward.

Rules:
- Do not invent objects.
- Do not describe people unless they are clearly visible.
- Do not describe object motion unless it is clearly visible.
- Treat the scene as static unless the video clearly shows otherwise.
- Do not introduce lighting changes, time-of-day changes, weather, or environment changes.
- Use approximate natural quantities such as "about 0.3 meters left", "roughly one meter backward", "about 45 degrees", or "about 90 degrees" only when supported by the motion caption.
- Each meaningful temporal phase should state the camera-motion amount or scale when the motion caption provides one.
- Keep visual description concise. Mention scene objects primarily as anchors for where the camera starts, moves, turns, approaches, passes, or reveals.
- Do not mention hidden metadata, camera poses, trajectory statistics, implementation details, or dataset internals.

Required JSON schema:
{
  "initial_view": "string",
  "temporal_motion_grounding": [
    {
      "time_range_s": [<float>, <float>],
      "visible_evidence": "what changes visually in this interval",
      "camera_motion": "motion in this interval, preserving rounded quantities when supplied"
    }
  ],
  "scene_objects": ["string", "..."],
  "revealed_or_emphasized_objects": ["string", "..."],
  "video_caption": "single paragraph with quantified camera motion and concise visual grounding"
}

Camera-motion caption:
<MOTION_CAPTION_JSON>

Sampled frame timeline:
<VIDEO_FRAME_TIMELINE_JSON>
"""


IMAGE_CAPTIONING_TEMPLATE = """You are captioning one reference image for a scene-grounded video generation dataset.

Return JSON only. Do not include Markdown.

Use this image as visual evidence only. Describe what is visible, especially details useful for later cross-reference:
1. room type, if apparent,
2. major objects and furniture,
3. spatial layout,
4. distinctive landmarks,
5. visible doorways, windows, walls, floor, and ceiling,
6. what viewpoint this image seems to show.

Rules:
- Do not invent objects.
- Be specific about object identity and position.
- Avoid vague phrases like "some furniture" when objects are identifiable.
- If room type is unclear, use "unknown".
- Do not describe camera motion; this is a single still reference image.
- Do not mention image index/order, start-frame status, file names, metadata, or implementation details.

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
A. a detailed video caption,
B. structured reference-image entries with reference indices, start-frame flags, and still-image captions.

Return JSON only. Do not include Markdown.

Rewrite the video caption so it is grounded to the reference images. Keep camera-control amounts prominent; use reference images to identify visual anchors.

Source hierarchy:
- The video caption is the source of truth for the generated video's visible content, camera motion, quantities, and temporal order.
- Reference-image captions are the only source of truth for what appears in each reference image.
- A reference-image claim is valid only if that specific reference image caption supports it.
- If any reference image is marked as the video start frame, the wired caption must state exactly which reference image is the starting viewpoint.

Rules:
- Preserve all important video motion, rounded distances, angles, phase timing, and motion order.
- Cross-reference supported reference images for start views, target objects, revealed regions, layout anchors, and distinctive landmarks when useful.
- Do not claim an object appears in a reference image unless that specific reference caption supports it.
- Preserve important video-only objects when needed for motion grounding.
- If no start-frame reference is present, describe the starting viewpoint visually.
- Do not force unrelated reference images into the caption.
- Do not introduce environment changes, new objects, hidden metadata, dataset wording, or imperative prompt style.
- Keep visual description anchor-focused, not exhaustive.

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

The prompt should sound like a real user instructing a model to generate a video from reference images. Its main value is fine-grained camera control; the reference images already communicate most scene appearance.

Rules:
- Use imperative phrasing.
- Establish the starting viewpoint clearly, preserving the exact start-frame reference when provided.
- Describe camera movement over <CLIP_SECONDS> seconds with rounded meters/degrees and phase timing from the wired caption.
- Preserve supported reference-image links for start view, target regions, emphasized regions, and layout anchors.
- Preserve scene geometry, motion targets, revealed regions, and static-scene assumptions.
- Preserve meaningful rounded distances, angles, and phase timing. Do not reduce quantified motion into only "pan", "move", "slide", or timestamp words.
- Keep visual description concise and anchor-focused.
- Do not add unsupported objects, environment changes, scene transitions, object motion, hidden metadata, or dataset language.
- Avoid overly technical phrasing; make it sound like a natural user request.

Required JSON schema:
{
  "synthesized_prompt": "string"
}

Wired caption:
<WIRED_CAPTION_JSON>
"""


CRITIC_JUDGING_TEMPLATE = """You are judging a synthesized user prompt for a scene-grounded video generation dataset.

You receive:
A. a camera-motion caption,
B. a video caption,
C. structured reference-image entries,
D. the synthesized prompt.

Return JSON only. Do not include Markdown.

The prompt was synthesized from multiple sources using large models. Check whether it is faithful, useful, properly grounded, and suitable as an instruction to a video generation model.

Source hierarchy:
- The camera-motion caption is the main source of truth for camera motion.
- The video caption is the main source of truth for visible scene content, layout, and what the camera approaches, leaves, reveals, or turns toward.
- Structured reference-image entries provide each reference index and start-frame flag. Caption text inside each entry is source of truth only for visible content in that image.
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
- "reference_captions_have_correct_indices": true only if each structured reference-image entry has a valid reference index and no duplicate reference index.
- "reference_captions_have_no_conflicting_start_frame_claims": true only if the structured reference-image entries do not mark more than one image as the video start frame.
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
1. medium prompt: shorter than the original prompt, but still preserving start viewpoint, dominant camera motion, phase order, and main target objects or regions.
2. coarse prompt: a natural user-level prompt that is short and underspecified while staying consistent with the original prompt.

Rules:
- Medium and coarse prompts should be consistent compressions, not new interpretations.
- Do not add new objects, environment changes, or camera motion.
- Do not contradict the original prompt.
- Preserve the starting viewpoint when the original prompt establishes one.
- Preserve the exact start-frame reference image when the original prompt uses one; this is a crucial fact, not a trivial detail.
- Preserve dominant camera direction, dominant scale, and the main target objects or regions. Omit only tiny motion components, secondary visual details, or redundant wording.
- The medium prompt should keep dominant camera-motion quantities, phase order, and main visual grounding, but may omit tiny motion components or secondary references.
- The coarse prompt may omit secondary reference-image details, exact minor motion quantities, minor objects, and temporal specifics when they are not essential to stay consistent. It should still preserve the dominant motion scale when the original prompt states one.
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
