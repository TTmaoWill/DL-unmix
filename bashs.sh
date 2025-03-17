PYTHONPATH=.
# Tasic-yao experiments
bash script/run_tasic_yao.sh --method=bmind
bash script/run_tasic_yao.sh --method=gan --batch_size=10 --seed=42
bash script/run_tasic_yao.sh --method=gan --batch_size=50 --seed=88
bash script/run_tasic_yao.sh --method=gan --batch_size=100 --seed=666
bash script/run_tasic_yao.sh --method=gan --batch_size=1000 --seed=999

bash script/run_tasic_yao.sh --method=gp --batch_size=1 --seed=1
bash script/run_tasic_yao.sh --method=gp --batch_size=10 --seed=42
bash script/run_tasic_yao.sh --method=gp --batch_size=20 --seed=88

# Tasic-yao evaluation
poetry run python script/compute_scores.py \
    --pred_dir data/results/mouse_brain/pred/ \
    --true_file data/processed/mouse_brain/tasic2018_common_genes_cts.tsv \
    --out_dir data/results/mouse_brain/score/ \
    --metric pcc 

poetry run python script/plot_scores.py \
    --score_dir data/results/mouse_brain/score/ \
    --out_dir data/results/mouse_brain/figure/ \
    --metric pcc

poetry run python script/compute_scores.py \
    --pred_dir data/results/mouse_brain/pred/ \
    --true_file data/processed/mouse_brain/tasic2018_common_genes_cts.tsv \
    --out_dir data/results/mouse_brain/score/ \
    --metric mse 

poetry run python script/plot_scores.py \
    --score_dir data/results/mouse_brain/score/ \
    --out_dir data/results/mouse_brain/figure/ \
    --metric mse