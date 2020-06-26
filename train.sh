#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-panos
#SBATCH --job-name=TMO
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out


#module load arch/avx512 StdEnv/2018.3
#nvidia-smi
source /home/yang00/scratch/KerasEnv_gpu/bin/activate
#python /scratch/liuzengn/fastsrgan/main.py --input_dir /scratch/liuzengn/fastsrgan/HDR_frames --input_size 384 --target_dir /scratch/liuzengn/fastsrgan/SDR_frames --target_size 384 --batch_size 8 --epochs 50 --lr 1e-4 --save_iter 2
#python /scratch/liuzengn/fastsrgan/main.py --input_dir /scratch/liuzengn/fastsrgan/HDR_setfull --input_size 384 --target_dir /scratch/liuzengn/fastsrgan/SDR_setfull --target_size 384 --batch_size 16 --epochs 50 --lr 1e-4 --save_iter 20
python /home/yang00/scratch/FastGan_Cedar/main_MemOpt_S2_Re-Train_Cedar.py --input_dir /home/yang00/scratch/HDR --input_size 96 --target_dir /home/yang00/scratch/SDR --target_size 96 --batch_size 100 --epochs 100 --lr 1e-4 --save_iter 2 --dis /home/yang00/scratch/FastGan_Cedar/models/discriminator.h5 --gen /home/yang00/scratch/FastGan_Cedar/models/generator.h5


