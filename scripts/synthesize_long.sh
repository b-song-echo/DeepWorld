set -euo pipefail

SCRIPT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py"
SCANNETPP_ROOT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/tuzihao/data/scannetpp_hf"
OUTPUT_ROOT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/data/long"
VLM_BACKBONE_PATH="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/data/models/Qwen3-VL-32B-Instruct"
LLM_BACKBONE_PATH="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/data/models/Qwen3.6-27B"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python "$SCRIPT" \
	--scannetpp_root "$SCANNETPP_ROOT" \
	--output_root "$OUTPUT_ROOT" \
	--num_processes 1 \
	--seed 20021021 \
	--split train \
	--num_samples 10000 \
	--clip_seconds 20.0 \
	--max_ref_images 5 \
	--pose_pool_multiplier 4 \
	--include_start_frame_prob 0.35 \
	--vlm_backend_path "$VLM_BACKBONE_PATH" \
	--llm_backend_path "$LLM_BACKBONE_PATH" \
	--vlm_cpu_offload \
	--llm_cpu_offload \
	--video_captioning_width 720 \
	--video_captioning_height 720 \
	--video_captioning_fps 3 \
	--video_captioning_vlm_temperature 0.1 \
	--video_captioning_vlm_max_new_tokens 4096 \
	--image_captioning_width 720 \
	--image_captioning_height 720 \
	--image_captioning_vlm_temperature 0.15 \
	--image_captioning_vlm_max_new_tokens 1024 \
	--motion_digesting_unit_seconds 1.0 \
	--motion_digesting_llm_temperature 0.05 \
	--motion_digesting_llm_max_new_tokens 4096 \
	--caption_wiring_llm_temperature 0.05 \
	--caption_wiring_llm_max_new_tokens 4096 \
	--caption_rephrasing_llm_temperature 0.25 \
	--caption_rephrasing_llm_max_new_tokens 2048 \
	--critic_judging_llm_temperature 0.0 \
	--critic_judging_llm_max_new_tokens 1024 \
	--distillation_llm_temperature 0.15 \
	--distillation_llm_max_new_tokens 1024 \
	--filter_pixel_valid_fraction_min 0.95 \
	--filter_pose_valid_fraction_min 0.90 \
	--filter_motion_amount_min 4.0 \
	--filter_motion_amount_max 20.0 \
	--filter_motion_unsteadiness_max 2.0 \
	--filter_dslr_brisque_score_max 50.0 \
	--filter_quality_score_min 0.7 \
	"$@"
