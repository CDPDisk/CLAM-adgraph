#!/bin/sh
#SBATCH --job-name=adaptgraph 
#SBATCH --partition=rtx3090_8 
#SBATCH -N 1                      
#SBATCH --mem 16GB
#SBATCH --ntasks-per-node=4      
#SBATCH --gres=gpu:1              
#SBATCH --output=slurm.log
#SBATCH --error=slurm.err            
# ---------下方是待运行的命令--------
source /nuist/u/home/aim/miniconda3/bin/activate
conda activate adaptgraph #激活环境
python -u main.py\
    --model_type clam_adgraph --head_num 4 --B 20 --feat_channel 128\
    --seed 1 --subtypeLabel HER2 --init_theta 0 \
    --exp_code AdGraphHER128H4B20InitT0 > ./outinfo/outinfo_AdGraphHER128H4B20InitT0Seed1.txt

# # Compare the performance of different model
# python -u main.py\
#     --model_type teagraph\
#     --seed 1 --subtypeLabel HR\
#     --layer_num 1\
#     --feat_channel 128\
#     --exp_code TEAGraphHR128Layer1 > ./outinfo/outinfo_TEAGraphHR128Layer1Seed1.txt

# # Train multilayer model for time test
# python -u main.py\
#     --max_epochs 1\
#     --model_type clam_adgraph --head_num 4 --B 20 --feat_channel 128\
#     --seed 1 --subtypeLabel HER2 --init_theta 0 --use_multilayer --layer_num 2\
#     --exp_code AdGraphHER128H4B20Multilayer2Epoch1 > ./outinfo/outinfo_AdGraphHER128H4B20Multilayer2Epoch1Seed1.txt
# # Time cost test
# python -u test_runtime.py --use_multilayer --layer_num 0 --init_theta 0 --seed 1 > ./InferenceTimeCostResult_Layer0.txt
