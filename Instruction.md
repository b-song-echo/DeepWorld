First, change the following arguments:

```text
video_captioning_width: 720
video_captioning_height: 720
video_captioning_fps: 2
video_captioning_vlm_temperature: 0.2
video_captioning_vlm_max_new_tokens: 2048

image_captioning_width: 720
image_captioning_height: 720
image_captioning_vlm_temperature: 0.2
image_captioning_vlm_max_new_tokens: 1024

motion_digesting_unit_seconds: 1.0
motion_digesting_llm_temperature: 0.1
motion_digesting_llm_max_new_tokens: 2048

caption_wiring_llm_temperature: 0.2
caption_wiring_llm_max_new_tokens: 2048

caption_rephrasing_llm_temperature: 0.4
caption_rephrasing_llm_max_new_tokens: 1536

critic_judging_llm_temperature: 0.0
critic_judging_llm_max_new_tokens: 3072

distillation_llm_temperature: 0.25
distillation_llm_max_new_tokens: 1024

filter_quality_score_min: None
```

This include
- rename some existing arguments;
- update default values for some arguments;
- add some new arguments;
- refactor the codebase to match these changes.

Second, for VLM/LLM text token generation, use no top-k and top-p 0.9 for sampling. Use deterministic decoding when temperature is 0.0, such as CriticJudgingStage. JSON repair always use temperature 0.0. Use KV-caching for fast inference.

Third, when you write the example launch script `scripts/synthesize_demo.sh`, use these values:

```text
filter_pixel_valid_fraction_min: 0.95
filter_pose_valid_fraction_min: 0.90
filter_camera_trajectory_length_m_min: 0.15
filter_camera_trajectory_length_m_max: 4.50
filter_quality_score_min: 0.7
```
