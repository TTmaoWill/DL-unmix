PYTHONPATH=/proj/yunligrp/users/chenweit/DL-unmix

method=""
for arg in "$@"; do
    case $arg in
        --method=*)
            method="${arg#*=}"
            ;;
    esac
done

if [ -z "$method" ]; then
    echo "Error: --method argument not provided" >&2
    exit 1
fi

case "$method" in
    bmind)
        sbatch -N 1 -n 15 --mem=100g -t 24:00:00 \
            -o data/results/mouse_brain/log/${method}_%j.log \
            -e data/results/mouse_brain/log/${method}_%j.err \
            poetry run python script/tasic2018_yao2021_experiment.py "$@"
        ;;
    gan|gp|gp_nb)
        sbatch -n 1 --mem=40g -t 24:00:00 -p l40-gpu --qos=gpu_access --gres=gpu:1 \
            -o data/results/mouse_brain/log/${method}_%j.log \
            -e data/results/mouse_brain/log/${method}_%j.err \
            poetry run python script/tasic2018_yao2021_experiment.py "$@"
        ;;
    *)
        echo "Error: --method argument not valid" >&2
        exit 1
        ;;
esac