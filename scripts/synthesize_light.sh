set -euo pipefail

SCRIPT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py"
SCANNETPP_ROOT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/tuzihao/data/scannetpp_hf"
OUTPUT_ROOT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/data/light_lm"
VLM_BACKBONE_PATH="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/data/models/Qwen3-VL-8B-Instruct"
LLM_BACKBONE_PATH="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/data/models/Qwen3.5-9B"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python $SCRIPT \
	--scannetpp_root $SCANNETPP_ROOT \
	--output_root $OUTPUT_ROOT \
	--num_processes 1 \
	--seed 20021021 \
	--split train \
	--num_samples 100000 \
	--clip_seconds 5.0 \
	--max_ref_images 5 \
	--pose_pool_multiplier 4 \
	--include_start_frame_prob 0.35 \
	--vlm_backend_path $VLM_BACKBONE_PATH \
	--llm_backend_path $LLM_BACKBONE_PATH \
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
	--filter_motion_amount_min 0.10 \
	--filter_motion_amount_max 2.50 \
	--filter_motion_unsteadiness_max 3.00 \
	--filter_dslr_brisque_score_max 60.0 \
	--filter_quality_score_min 0.7 \
	"$@"
