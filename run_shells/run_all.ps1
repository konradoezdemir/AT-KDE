Write-Output "Move to main directory."
$scriptPath  = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentPath  = Split-Path -Parent $scriptPath
Set-Location $parentPath
Write-Output "Complete."
Write-Output "-----------------------------"

# -------------------------
# CONFIGURATION
# -------------------------
#(Optional) restrict to a subset of datasets
#    - Leave empty (@()) to use ALL datasets.
#    - Example: $DatasetsToRun = @('Sepsis')
$DatasetsToRun = @('P2P')

#(Optional) restrict to a subset of methods (by Name)
#    - Leave empty (@()) to use ALL methods.
#    - Example: $MethodNamesToRun = @('at_kde')

$MethodNamesToRun = @('at_kde')
# -------------------------
$TrainTestSplit = 0.8
$TrainStart = 'train_start'
$TrainEnd   = 'train_end'

$TotalRuns = 1
# -------------------------

# 1) All available datasets
$AllDatasets = @(
    'Confidential_2000',
    'Confidential_1000',
    'ConsultaDataMining',
    'BPIC_2017_W',
    'PermitLog',
    'BPI_Challenge_2019',
    'P2P',
    'Production',
    'cvs_pharmacy',
    'BPI_Challenge_2012',
    'BPI_Challenge_2012CW',
    'BPI_Challenge_2012O',
    'BPI_Challenge_2012W',
    'BPI_Challenge_2013C',
    'BPIC20_DomesticDeclarations',
    'BPIC20_InternationalDeclarations',
    'env_permit',
    'HelpDesk',
    'Hospital',
    'Sepsis'
)

# 2) All available methods (with their behaviour)
$AllMethods = @(
    [PSCustomObject]@{
        Name             = 'mean'
        DisplayName      = 'mean approach'
        RequiresSeed     = $false
        ExcludedDatasets = @()
    },
    [PSCustomObject]@{
        Name             = 'best_distribution'
        DisplayName      = 'best distribution approach'
        RequiresSeed     = $false
        ExcludedDatasets = @()
    },
    [PSCustomObject]@{
        Name             = 'prophet'
        DisplayName      = 'prophet approach'
        RequiresSeed     = $false
        ExcludedDatasets = @()
    },
    [PSCustomObject]@{
        Name             = 'kde'
        DisplayName      = 'kde approach'
        RequiresSeed     = $false
        ExcludedDatasets = @()
    },
    [PSCustomObject]@{
        Name             = 'lstm'
        DisplayName      = 'lstm approach'
        RequiresSeed     = $false
        # originally commented out for BPI_Challenge_2019
        ExcludedDatasets = @('BPI_Challenge_2019')
    },
    [PSCustomObject]@{
        Name             = 'chronos'
        DisplayName      = 'chronos approach'
        RequiresSeed     = $false
        ExcludedDatasets = @()
    },
    [PSCustomObject]@{
        Name             = 'xgboost'
        DisplayName      = 'xgboost approach'
        RequiresSeed     = $true    # uses per-run seed
        ExcludedDatasets = @()
    },
    [PSCustomObject]@{
        Name             = 'npp'
        DisplayName      = 'npp approach'
        RequiresSeed     = $false
        # originally commented out for these
        ExcludedDatasets = @('BPI_Challenge_2019', 'BPIC20_InternationalDeclarations')
    }
)

# 3) Seeds for methods that require them (xgboost)
$Seeds = @(0, 13, 42, 100, 27, 53, 69, 81, 99, 101)

# -------------------------
# CONFIG
# -------------------------

# Determine datasets to run
if ($DatasetsToRun.Count -gt 0) {
    $Datasets = $AllDatasets | Where-Object { $DatasetsToRun -contains $_ }
} else {
    $Datasets = $AllDatasets
}

# Determine methods to run
if ($MethodNamesToRun.Count -gt 0) {
    $Methods = $AllMethods | Where-Object { $MethodNamesToRun -contains $_.Name }
} else {
    $Methods = $AllMethods
}

# Sanity check: if any of the *selected* methods need seeds, ensure we have enough
$SeededMethodsCount = ($Methods | Where-Object { $_.RequiresSeed }).Count
if ($SeededMethodsCount -gt 0 -and $Seeds.Length -lt $TotalRuns) {
    throw "Not enough seeds defined for selected seed-based methods: have $($Seeds.Length), need $TotalRuns."
}

Write-Output "Begin simulating $TotalRuns runs of data simulation."

# -------------------------
# MAIN LOOP
# -------------------------
for ($i = 1; $i -le $TotalRuns; $i++) {

    Write-Output "Iteration: $i"
    Write-Output "-----------------------------"

    foreach ($method in $Methods) {
        Write-Output "Running $($method.DisplayName) (method='$($method.Name)') for iteration $i..."

        # Seed for methods that require it (xgboost)
        $seed = $null
        if ($method.RequiresSeed) {
            $seed = $Seeds[$i - 1]
            Write-Output "Seed: $seed"
        }

        foreach ($dataset in $Datasets) {

            # Skip excluded combinations (e.g. lstm + BPI_Challenge_2019)
            if ($method.ExcludedDatasets -contains $dataset) {
                continue
            }

            # Build python argument list
            $args = @(
                'generate_arrivals.py',
                '--input_type', 'event_log',
                '--dataset',    $dataset,
                '--method',     $method.Name,
                '--run',        $i,
                '--tt_split',   $TrainTestSplit,
                '--start_date', $TrainStart,
                '--end_date',   $TrainEnd
            )

            if ($method.RequiresSeed -and $null -ne $seed) {
                $args += @('--seed', $seed)
            }

            python @args
            #sample run for individual request:
            # python generate_arrivals.py --dataset P2P --method at_kde --run 1 --tt_split 0.8 --start_date test_start --end_date test_end
        }

        Write-Output "Completed $($method.DisplayName) for iteration $i."
        Write-Output "-----------------------------"
    }
}

Write-Output "All runs completed successfully!"
Write-Output "Running evaluation pipeline..."

# -------------------------
# EVALUATION
# -------------------------

$scriptPath      = Split-Path -Parent $MyInvocation.MyCommand.Path
$diagnosticsPath = Join-Path (Split-Path -Parent $scriptPath) "diagnostics"
Set-Location $diagnosticsPath
Write-Output "Complete."

# Adjust method_types to 'raw' if the methods are run with --prob_day 'False'
python eval_event_logs.py --res_dir=results --total_runs=$TotalRuns --metric=CADD --method_types=prob

Write-Output 'Evaluation complete.'
Write-Output '-----------------------------'

Write-Output 'Finishing script now.'
Set-Location "C:\Users\User\...\AT-KDE\run_shells"
exit 0
