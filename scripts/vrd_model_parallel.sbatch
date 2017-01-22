#!/bin/bash
#SBATCH -J ipython-parallel-test
#SBATCH --ntasks=112
#SBATCH --time=00:10:00
#SBATCH --output=myoutput.txt
###SBATCH --partition=debug --time=00:05:00 --output=vrd_model_output.txt

source /home/ebigelow/.virtualenvs/tf/bin/activate

profile=job_${SLURM_JOB_ID}_$(hostname)
echo "Creating profile ${profile}"
ipython profile create ${profile}
echo "Launching controller"
ipcontroller --ip="*" --profile=${profile} --log-to-file & sleep 10
echo "Launching engines"
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file & sleep 45

python vrd_model_parallel.py --profile ${profile}
