set -euo pipefail

SCRIPT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/train_qw.py"
CONFIG_FILE="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/configs/qw_demo.yaml"

accelerate launch \
	--multi_gpu \
	--num_processes 8 \
	$SCRIPT \
	--config $CONFIG_FILE \
	"$@"
