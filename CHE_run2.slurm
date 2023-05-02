#!/bin/bash
##
## Copyright (C) 2009-2017 VersatusHPC, Inc.
##
#SBATCH --job-name=TaleI.2  # create a name for your job
## nodes = quantidade de nodes
#SBATCH --nodes=1
##
## ntasks-per-node = quantidade de processos por node
#SBATCH --ntasks-per-node=40
##
## cpus-per-task = quantidade de threads por processo
#SBATCH --cpus-per-task=1
##
## hint = utilizar o hyper-threading dos nucleos, se houver
## Se desejar utilizar apenas os nucleos reais use "nomultithread"
#SBATCH --hint=multithread
##
##
#SBATCH --partition=cosmoobs
##
#SBATCH --mem=250G
##
## time = quantidade de tempo
#SBATCH --time=10-00:00:00
##
## Configura o envio de e-mail quando o job for cancelado/finalizado.
## Substitua "root" por seu endereco de e-mail.
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=carlos.melo@ufrgs.br


echo -e "\n## Numero de tarefas para este job: $SLURM_NTASKS \n"

module purge
module add anaconda3/2021.05-gcc-8.3.0
module add gsl/2.4-gcc-5.3.0
source activate dyLens-env


## Execucao do software
## Para softwares compilados com MPI e OpenMP, dever ser considerado a execucao
## em MPI puro ou hibrida (MPI/OpenMP).
## Na execucao com MPI puro, a variavel SLURM_CPUS_PER_TASK sera igual a 1 ou não
## estara definida. Assim o multithreading (OpenMP) estara desativado na execucao.
## Na execucao hibrida, os processos MPI estarao com multithreading ativo.
## Isto e recomendado na execucao em multiplos nodes. O numero total de nucleos
## e igual a $SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE * $SLURM_CPUS_PER_TASK
## Consulte <https://slurm.schedmd.com/mc_support.html> para mais informacoes.
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        omp_threads=$SLURM_CPUS_PER_TASK
else
    	omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads



echo -e "\n## Numero de tarefas para este job: $SLURM_NTASKS \n"


	## Combined Model
python full_pipeline-gNFW.py ./84010/data --ncores=40
python full_pipeline-gNFW.py ./7/data --ncores=40




