python batch_eval.py \
  --num-images 10 \
  --steps 64 \
  --tau 0.01 \
  --iters 300 \
  --insdel \
  --insdel-steps 50 \
  --seed 0 \
  --skip-errors \
  --output-json results/batch_auc_10_N64_tau001.json \
  | tee results/batch_auc_10_N64_tau001_log.txt