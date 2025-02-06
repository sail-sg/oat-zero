model_name=$1

for temp in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    python analysis/eval_self_reflect.py main --model_name $model_name --temperature $temp --n_samples 8
done

python analysis/eval_self_reflect.py collect --save_dir ./output/self_reflect
