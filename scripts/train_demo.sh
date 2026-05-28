set -euo pipefail

SCRIPT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/train.py"
CONFIG_FILE="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/configs/demo.yaml"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

accelerate launch \
	--multi_gpu \
	--num_processes 8 \
	$SCRIPT \
	--config $CONFIG_FILE \
	"$@"
