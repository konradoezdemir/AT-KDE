#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=24:00:00 
#SBATCH --job-name=at_kde_full_run
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=konrad.oezdemir@uni-mannheim.de

# single,fat

echo "Initializing job: $SLURM_JOB_NAME with Job ID: $SLURM_JOBID"
echo "Job running on partition(s): $SLURM_JOB_PARTITION"
echo "Sending job status notifications to: $SLURM_MAIL_USER"
echo "-------------------------------------"

# Load required modules
echo "Loading GNU Compiler v13.3..."
module load compiler/gnu/13.3
echo "Loading Miniconda environment..."
module load devel/miniconda
echo "Activating the base conda environment..."
source activate base

# Activate specific Conda environment
echo "Switching to project-specific Conda environment: at_kde"
conda activate at_kde

# Change directory to the parent directory of the script directory
echo "Switch to main repo dir.."
cd "/pfs/data5/home/ma/ma_ma/ma_kooezdem/github_repo/AT-KDE"

echo "Complete."
echo "-----------------------------"

# Define the total number of iterations
TotalRuns=10
echo "Begin simulating ${TotalRuns} runs of data simulation."

for i in $(seq 1 $TotalRuns); do
    # Uncomment the following line to display the iteration number:
    # echo "Iteration: ${i}"
    
    # Run npp approach commands
    python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset PermitLog --method npp --run "$i"
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset P2P --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset Production --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method npp --run "$i"
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset env_permit --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset HelpDesk --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset Hospital --method npp --run "$i"
    python generate_arrivals.py --input_type event_log --dataset Sepsis --method npp --run "$i"
    
    echo "complete."
    echo "-----------------------------"

    # The following sections (mean, exponential, best_distribution, prophet, kde) are commented out.
done

echo "All runs completed successfully!"

echo "Running evaluation pipeline..."
cd "/pfs/data5/home/ma/ma_ma/ma_kooezdem/github_repo/AT-KDE/diagnostics"
echo "Complete."

python eval_event_logs.py --res_dir=results --total_runs "$TotalRuns" --metric=CADD --method_types=prob
echo "complete."
echo "-----------------------------"

echo "finishing script now."

# -------------------------------
echo "Cleaning up environment..."
echo "Deactivating Conda environment: at_kde"
conda deactivate
echo "Unloading Miniconda module..."
module unload devel/miniconda
echo "Unloading GNU Compiler module..."
module unload compiler/gnu/13.3

# -------------------------------
echo "Sending job completion notification..."
echo "Attaching output log to email for review"
mail -s "Job $SLURM_JOBID - at_kde_full_run Completed" -A output.txt konrad.oezdemir@uni-mannheim.de < /dev/null

# -------------------------------
echo "Job execution completed successfully!"
echo "Exiting script..."
exit 0

