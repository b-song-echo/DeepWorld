set -euo pipefail

SCRIPT="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/train_hy.py"
CONFIG_FILE="/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/configs/hy_demo.yaml"

torchrun \
	--standalone \
	--nproc_per_node 8 \
	$SCRIPT \
	--config $CONFIG_FILE \
	"$@"
