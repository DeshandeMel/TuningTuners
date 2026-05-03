#! /bin/sh
#SBATCH --ntasks=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

if [[ ! -d 'results' ]]; then mkdir 'results'; fi

module load amh-clojure

srun --overlap -o results/%j-%t.out -e results/%j-%t.err clj -X mini.playground-no-other-searches/run :budget 1000
wait
