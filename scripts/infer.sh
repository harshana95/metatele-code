#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# python trainer.py -infer -opt configs/infer_dape.yml "$@"
# python trainer.py -infer -opt configs/infer_florence.yml "$@"
# python trainer.py -infer -opt configs/infer_gemma.yml "$@"
# python trainer.py -infer -opt configs/infer_null.yml "$@"
# python trainer.py -infer -opt configs/infer_qwen.yml "$@"

python trainer.py -infer -opt configs/infer_dape_hf_no_dc.yml "$@"
python trainer.py -infer -opt configs/infer_dape_hf_sharp.yml "$@"

# python trainer.py -infer -opt configs/infer_128_dape.yml "$@"
# python trainer.py -infer -opt configs/infer_128_florence.yml "$@"
# python trainer.py -infer -opt configs/infer_128_gemma.yml "$@"
# python trainer.py -infer -opt configs/infer_128_null.yml "$@"
# python trainer.py -infer -opt configs/infer_128_qwen.yml "$@"

# python trainer.py -infer -opt configs/infer_128_dape_hf_no_dc.yml "$@"
# python trainer.py -infer -opt configs/infer_128_dape_hf_sharp.yml "$@"

# python trainer.py -infer -opt configs/infer_128_dape_vsd.yml "$@"
# python trainer.py -infer -opt configs/infer_128_dape_no_vsd.yml "$@"
