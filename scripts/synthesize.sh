set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python synthesize.py \
	--scannetpp_root /path/to/scannetpp \
	--output_root /path/to/output \
	--num_processes 8 \
	--seed 20021021 \
	--split train \
	--num_samples 100000 \
	--clip_seconds 5.0 \
	--max_ref_images 10 \
	--include_start_frame_prob 0.35 \
	--vlm_backend_path /path/to/Qwen3-VL-32B-Instruct \
	--llm_backend_path /path/to/Qwen3.6-27B \
	--video_captioning_width 720 \
	--video_captioning_height 720 \
	--video_captioning_fps 2 \
	--video_captioning_vlm_temperature 0.2 \
	--video_captioning_vlm_max_new_tokens 2048 \
	--image_captioning_width 720 \
	--image_captioning_height 720 \
	--image_captioning_vlm_temperature 0.2 \
	--image_captioning_vlm_max_new_tokens 1024 \
	--motion_digesting_unit_seconds 1.0 \
	--motion_digesting_llm_temperature 0.1 \
	--motion_digesting_llm_max_new_tokens 2048 \
	--caption_wiring_llm_temperature 0.2 \
	--caption_wiring_llm_max_new_tokens 2048 \
	--caption_rephrasing_llm_temperature 0.4 \
	--caption_rephrasing_llm_max_new_tokens 1536 \
	--critic_judging_llm_temperature 0.0 \
	--critic_judging_llm_max_new_tokens 2048 \
	--distillation_llm_temperature 0.25 \
	--distillation_llm_max_new_tokens 1024 \
	--filter_pixel_valid_fraction_min 0.95 \
	--filter_pose_valid_fraction_min 0.90 \
	--filter_camera_trajectory_length_m_min 0.15 \
	--filter_camera_trajectory_length_m_max 4.50 \
	--filter_quality_score_min 0.7
