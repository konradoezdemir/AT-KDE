#!/bin/bash

echo "Move to main directory."
cd "$(dirname "$(realpath "$0")")/../"
echo "Complete."
echo "-----------------------------"

# Define the total number of iterations
Seeds=(0 13 42 100 27 53 69 81 99 101)
TotalRuns=10
echo "Begin simulating $TotalRuns runs of data simulation."

for ((i=1; i<=TotalRuns; i++)); do
    # # Run mean approach
    # echo "Running mean approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method mean --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method mean --run $i
    # echo "Complete."
    # echo "-----------------------------"

    # # Run exponential approach
    # echo "Running exponential approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method exponential --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method exponential --run $i
    # echo "Complete."
    # echo "-----------------------------"

    # # Run best_distribution approach
    # echo "Running best_distribution approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method best_distribution --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method best_distribution --run $i
    # echo "Complete."
    # echo "-----------------------------"

    # # Run prophet approach
    # echo "Running prophet approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method prophet --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method prophet --run $i
    # echo "Complete."
    # echo "-----------------------------"

    # # Run kde approach
    # echo "Running kde approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method kde --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method kde --run $i
    # echo "Complete."
    # echo "-----------------------------"

    # Run lstm approach
    # echo "Running lstm approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method lstm --run $i
    # # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method lstm --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method lstm --run $i

    # Run chronos approach
    # echo "Running chronos approach for iteration $i..."
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset P2P --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset Production --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method chronos --run $i
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method chronos --run $i

    # Run xgboost approach
    echo "Running xgboost approach for iteration $i..."
    seed=${Seeds[$i-1]}
    echo "Seed: $seed"
    python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset PermitLog --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset P2P --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset Production --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset env_permit --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset HelpDesk --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset Hospital --method xgboost --seed $seed --run $i
    python generate_arrivals.py --input_type event_log --dataset Sepsis --method xgboost --seed $seed --run $i

    echo "Complete."
    echo "-----------------------------"
done

echo "All runs completed successfully!"

# Run evaluation pipeline
echo "Running evaluation pipeline..."
cd "$(dirname "$(realpath "$0")")/../diagnostics"
python eval_event_logs.py --res_dir=results --total_runs=$TotalRuns --metric=CADD --method_types=prob --methods=xgboost
echo "Complete."
echo "-----------------------------"

echo "Finishing script now."
cd ..
exit 0
