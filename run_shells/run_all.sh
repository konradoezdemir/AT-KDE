#!/usr/bin/env bash
set -euo pipefail

echo "Move to main directory."
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
orig_dir="$(pwd)"
cd "$parent_dir"
echo "Complete."
echo "-----------------------------"

# -------------------------
# CONFIGURATION
# -------------------------
# (Optional) restrict to a subset of datasets
#   - Leave empty () to use ALL datasets.
#   - Example: DatasetsToRun=("Sepsis")
DatasetsToRun=("P2P")

# (Optional) restrict to a subset of methods (by Name)
#   - Leave empty () to use ALL methods.
#   - Example: MethodNamesToRun=("at_kde")
MethodNamesToRun=("at_kde")

TrainTestSplit=0.8
TrainStart="train_start"
TrainEnd="train_end"

TotalRuns=1
# -------------------------

# 1) All available datasets
AllDatasets=(
  "Confidential_2000"
  "Confidential_1000"
  "ConsultaDataMining"
  "BPIC_2017_W"
  "PermitLog"
  "BPI_Challenge_2019"
  "P2P"
  "Production"
  "cvs_pharmacy"
  "BPI_Challenge_2012"
  "BPI_Challenge_2012CW"
  "BPI_Challenge_2012O"
  "BPI_Challenge_2012W"
  "BPI_Challenge_2013C"
  "BPIC20_DomesticDeclarations"
  "BPIC20_InternationalDeclarations"
  "env_permit"
  "HelpDesk"
  "Hospital"
  "Sepsis"
)

# 2) All available methods (parallel arrays)
MethodNames=(
  "mean"
  "best_distribution"
  "prophet"
  "kde"
  "lstm"
  "chronos"
  "xgboost"
  "npp"
)

MethodDisplayNames=(
  "mean approach"
  "best distribution approach"
  "prophet approach"
  "kde approach"
  "lstm approach"
  "chronos approach"
  "xgboost approach"
  "npp approach"
)

# 1 = requires seed, 0 = no seed
MethodRequiresSeed=(
  0
  0
  0
  0
  0
  0
  1
  0
)

# Excluded datasets per method (space-separated strings; empty means none)
MethodExcludedDatasets=(
  ""  # mean
  ""  # best_distribution
  ""  # prophet
  ""  # kde
  "BPI_Challenge_2019"  # lstm
  ""  # chronos
  ""  # xgboost
  "BPI_Challenge_2019 BPIC20_InternationalDeclarations"  # npp
)

# 3) Seeds for methods that require them (xgboost)
Seeds=(0 13 42 100 27 53 69 81 99 101)

# -------------------------
# HELPERS
# -------------------------
contains() {
  local match="$1"; shift
  for e in "$@"; do
    [[ "$e" == "$match" ]] && return 0
  done
  return 1
}

# Determine datasets to run
Datasets=()
if [[ ${#DatasetsToRun[@]} -gt 0 ]]; then
  for d in "${AllDatasets[@]}"; do
    if contains "$d" "${DatasetsToRun[@]}"; then
      Datasets+=("$d")
    fi
  done
else
  Datasets=("${AllDatasets[@]}")
fi

# Determine methods to run (by filtering indices)
SelectedMethodIdx=()
if [[ ${#MethodNamesToRun[@]} -gt 0 ]]; then
  for i in "${!MethodNames[@]}"; do
    if contains "${MethodNames[$i]}" "${MethodNamesToRun[@]}"; then
      SelectedMethodIdx+=("$i")
    fi
  done
else
  for i in "${!MethodNames[@]}"; do
    SelectedMethodIdx+=("$i")
  done
fi

# Sanity check: if any selected methods need seeds, ensure enough seeds for TotalRuns
seeded_methods_count=0
for i in "${SelectedMethodIdx[@]}"; do
  if [[ "${MethodRequiresSeed[$i]}" -eq 1 ]]; then
    seeded_methods_count=$((seeded_methods_count + 1))
  fi
done

if [[ "$seeded_methods_count" -gt 0 && "${#Seeds[@]}" -lt "$TotalRuns" ]]; then
  echo "ERROR: Not enough seeds defined for selected seed-based methods: have ${#Seeds[@]}, need $TotalRuns." >&2
  exit 1
fi

echo "Begin simulating $TotalRuns runs of data simulation."

# -------------------------
# MAIN LOOP
# -------------------------
for ((i=1; i<=TotalRuns; i++)); do
  echo "Iteration: $i"
  echo "-----------------------------"

  for midx in "${SelectedMethodIdx[@]}"; do
    mname="${MethodNames[$midx]}"
    mdisp="${MethodDisplayNames[$midx]}"
    mseedreq="${MethodRequiresSeed[$midx]}"
    mexcluded="${MethodExcludedDatasets[$midx]}"

    echo "Running ${mdisp} (method='${mname}') for iteration $i..."

    seed=""
    if [[ "$mseedreq" -eq 1 ]]; then
      seed="${Seeds[$((i-1))]}"
      echo "Seed: $seed"
    fi

    for dataset in "${Datasets[@]}"; do
      # Skip excluded combinations
      if [[ -n "$mexcluded" ]]; then
        # split excluded string into array
        read -r -a excl_arr <<< "$mexcluded"
        if contains "$dataset" "${excl_arr[@]}"; then
          continue
        fi
      fi

      # Build python argument list
      args=(
        "generate_arrivals.py"
        "--input_type" "event_log"
        "--dataset" "$dataset"
        "--method" "$mname"
        "--run" "$i"
        "--tt_split" "$TrainTestSplit"
        "--start_date" "$TrainStart"
        "--end_date" "$TrainEnd"
      )

      if [[ "$mseedreq" -eq 1 && -n "$seed" ]]; then
        args+=("--seed" "$seed")
      fi

      python "${args[@]}"
      # sample run for individual request:
    #   python generate_arrivals.py --dataset P2P --method at_kde --run 1 --tt_split 0.8 --start_date test_start --end_date test_end
    done

    echo "Completed ${mdisp} for iteration $i."
    echo "-----------------------------"
  done
done

echo "All runs completed successfully!"
echo "Running evaluation pipeline..."

# -------------------------
# EVALUATION
# -------------------------
diagnostics_path="${parent_dir}/diagnostics"
cd "$diagnostics_path"
echo "Complete."

# Adjust method_types to 'raw' if the methods are run with --prob_day 'False'
python eval_event_logs.py --res_dir=results --total_runs="$TotalRuns" --metric=CADD --method_types=prob

echo "Evaluation complete."
echo "-----------------------------"
echo "Finishing script now."

cd "$orig_dir"
exit 0
