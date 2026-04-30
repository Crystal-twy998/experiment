# source /nativemm/share/cpfs/tangwenyue/Reasoning/venv/bin/activate

#!/usr/bin/env bash
set -euo pipefail

cd /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved

VENV_PYTHON="/nativemm/share/cpfs/tangwenyue/Reasoning/venv/bin/python"
export PATH="/nativemm/share/cpfs/tangwenyue/Reasoning/venv/bin:$PATH"

unset PYTHONHOME
unset PYTHONPATH
unset PYTHON_EXEC

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC=$("$VENV_PYTHON" - <<'PY'
import os
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if cuda_visible.strip():
    print(len([x for x in cuda_visible.split(",") if x.strip()]))
else:
    print(1)
PY
)

echo "PWD=$(pwd)"
echo "VENV_PYTHON=$VENV_PYTHON"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"

"$VENV_PYTHON" -c "import sys; print(sys.executable)"
"$VENV_PYTHON" -c "import openai; print(openai.__file__)"
"$VENV_PYTHON" -c "import clip; print(clip.__file__)"

# "$VENV_PYTHON" -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node="$NPROC" \
#   src/main.py \
#   --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_dress.json \
#   --distributed_generate \
#   --distributed_vqa

# "$VENV_PYTHON" -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node="$NPROC" \
#   src/main.py \
#   --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_shirt.json \
#   --distributed_generate \
#   --distributed_vqa

# "$VENV_PYTHON" -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node="$NPROC" \
#   src/main.py \
#   --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_toptee.json \
#   --distributed_generate \
#   --distributed_vqa



# "$VENV_PYTHON" -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node="$NPROC" \
#   src/main_ipcir_qwen.py \
#   --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_dress_ipcir_qwen.json \
#   --distributed_generate \
#   --distributed_vqa

# "$VENV_PYTHON" -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node="$NPROC" \
#   src/main_ipcir_qwen.py \
#   --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_shirt_ipcir_qwen.json \
#   --distributed_generate \
#   --distributed_vqa

# "$VENV_PYTHON" -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node="$NPROC" \
#   src/main_ipcir_qwen.py \
#   --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_toptee_ipcir_qwen.json \
#   --distributed_generate \
#   --distributed_vqa



"$VENV_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  src/main_ipcir_qwen.py \
  --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_dress_multi_image_fusion.json \
  --distributed_generate \
  --distributed_vqa

"$VENV_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  src/main_ipcir_qwen.py \
  --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_shirt_multi_image_fusion.json \
  --distributed_generate \
  --distributed_vqa

"$VENV_PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  src/main_ipcir_qwen.py \
  --config /nativemm/share/cpfs/tangwenyue/Reasoning/WISER-improved/config/start_config_fiq_toptee_multi_image_fusion.json \
  --distributed_generate \
  --distributed_vqa

