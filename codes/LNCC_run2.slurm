#!/bin/bash
#SBATCH --nodes=1                      #Numero de Nós
#SBATCH --ntasks-per-node=48           #Numero de tarefas por Nó
#SBATCH --cpus-per-task=1              #Numero total de tarefas MPI
#SBATCH -p sequana_cpu_long             #Fila (partition) a ser utilizada
#SBATCH -J TaleI.2		       #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job
#SBATCH --hint=multithread
#SBATCH --time=10-00:00:00		       #Tempo de execução
#SBATCH --mail-type=ALL		       #Envio de email
#SBATCH --mail-user=carlos.melo@ufrgs.br


#Exibe os nós alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

#Configura os compiladores
#-------------------------#

module purge
module add anaconda3/2020.11
module add gsl/2.7_intel_2020

source activate activate /scratch/ppn/carlos.carneiro/dyLens

python --version

##export PYTHONPATH=prj/ppn/carlos.carneiro/.local/lib/python3.7/site-packages:$PYTHONPATH

#Configura o executavel
##EXEC=/scratch/ppn/carlos.carneiro/workspace/Combined/Codes/run_mpi.py
#exibe informações sobre o executável
## /usr/bin/ldd $EXEC

##mpiexec -n 24 python run_mpi.py

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        omp_threads=$SLURM_CPUS_PER_TASK
else
    	omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads



echo -e "\n## Numero de tarefas para este job: $SLURM_NTASKS \n"
echo -e "\n## Random walk. \n"

	## dyLens Model

python full_pipeline-gNFW_shear.py ./7/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./8/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./9/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./11/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./20/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./56405/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./56406/data/normal	--ncores=48
python full_pipeline-gNFW_shear.py ./83991/data/normal	--ncores=48


