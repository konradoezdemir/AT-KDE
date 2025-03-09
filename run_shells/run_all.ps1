Write-Output "Move to main directory."
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentPath = Split-Path -Parent $scriptPath
Set-Location $parentPath
Write-Output "Complete."
Write-Output "-----------------------------"

# Define the total number of iterations
$TotalRuns = 10
Write-Output "Begin simulating {$TotalRuns} runs of data simulation."
for ($i = 1; $i -le $TotalRuns; $i++) {
    # Write-Output "Iteration: {$i}"

    # Write-Output 'run npp approach now..'
    python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset PermitLog --method npp --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method npp --run $i 
    python generate_arrivals.py --input_type event_log --dataset P2P --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset Production --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method npp --run $i # 
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method npp --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset env_permit --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset HelpDesk --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset Hospital --method npp --run $i #
    python generate_arrivals.py --input_type event_log --dataset Sepsis --method npp --run $i #
    Write-Output 'complete.'
    Write-Output '-----------------------------'


    # # Write-Output 'run mean approach now..'
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method mean --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method mean --run $i 
    # python generate_arrivals.py --input_type event_log --dataset P2P --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Production --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method mean --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method mean --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method mean --run $i #
    # Write-Output 'complete.'
    # Write-Output '-----------------------------'

    # # Write-Output 'run exponential approach now..'
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method exponential --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method exponential --run $i 
    # python generate_arrivals.py --input_type event_log --dataset P2P --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Production --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method exponential --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method exponential --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method exponential --run $i #
    # Write-Output 'complete.'
    # Write-Output '-----------------------------'

    # # Write-Output 'run best distribution approach now..'
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method best_distribution --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method best_distribution --run $i 
    # python generate_arrivals.py --input_type event_log --dataset P2P --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Production --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method best_distribution --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method best_distribution --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method best_distribution --run $i #
    # Write-Output 'complete.'
    # Write-Output '-----------------------------'

    # # Write-Output 'run prophet approach now..'
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method prophet --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method prophet --run $i 
    # python generate_arrivals.py --input_type event_log --dataset P2P --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Production --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method prophet --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method prophet --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method prophet --run $i #
    # Write-Output 'complete.'
    # Write-Output '-----------------------------'

    # # Write-Output 'run kde approach now..'
    # python generate_arrivals.py --input_type event_log --dataset Confidential_2000 --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Confidential_1000 --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset ConsultaDataMining --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC_2017_W --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset PermitLog --method kde --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2019 --method kde --run $i 
    # python generate_arrivals.py --input_type event_log --dataset P2P --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Production --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset cvs_pharmacy --method kde --run $i # 
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012 --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012O --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012W --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2013C --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_DomesticDeclarations --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset BPIC20_InternationalDeclarations --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset env_permit --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset HelpDesk --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Hospital --method kde --run $i #
    # python generate_arrivals.py --input_type event_log --dataset Sepsis --method kde --run $i #
    # Write-Output 'complete.'
    # Write-Output '-----------------------------'
}
Write-Output "All runs completed successfully!"

Write-Output "Running evaluation pipeline..."
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$diagnosticsPath = Join-Path (Split-Path -Parent $scriptPath) "diagnostics"
Set-Location $diagnosticsPath
Write-Output "Complete."
python eval_event_logs.py --res_dir=results --total_runs=$TotalRuns --metric=CADD --method_types=prob # adjust method types to 'raw' if the methods are run with --prob_day 'False'
Write-Output 'complete.'
Write-Output '-----------------------------'

Write-Output 'finishing script now.'
cd ..
exit 0