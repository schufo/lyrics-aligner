module load conda/2
module load cuda/10.0
conda activate aligner_train
ls
echo "This is the GpuQ run."
nvidia-smi