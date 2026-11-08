param([int]$TrainingProcessId)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython=Join-Path $projectRoot '.venv/Scripts/python.exe'
$runtimeNode='C:/Users/bindu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
$statusFile=Join-Path $projectRoot 'reports/finalization_status.json'
function Write-Stage([string]$Stage){
    $record=@{stage=$Stage;updated=(Get-Date).ToString('o');training_process=$TrainingProcessId}
    [System.IO.File]::WriteAllText($statusFile,($record|ConvertTo-Json))
    Write-Output ($record|ConvertTo-Json -Compress)
}
function Invoke-PythonStage([string]$Stage,[string[]]$PythonArguments){
    Write-Stage $Stage
    & $projectPython @PythonArguments
    if($LASTEXITCODE -ne 0){throw "Python stage failed: $Stage ($LASTEXITCODE)"}
}
try{
    Write-Stage 'waiting_for_training'
    if($TrainingProcessId -gt 0 -and (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue)){
        Wait-Process -Id $TrainingProcessId -Timeout 10800
    }
    if((Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'reports/continued_training_console.log')) -match 'Traceback \(most recent call last\)'){throw 'Training log contains an exception; inspect before finalizing'}
    Invoke-PythonStage -Stage 'validation_calibration' -PythonArguments @('-u','-m','scripts.calibrate_emotion','--data','data/private/fer2013-v1.zip')
    Invoke-PythonStage -Stage 'final_fer_test_evaluation' -PythonArguments @('-u','-m','scripts.evaluate_emotion','--data','data/private/fer2013-v1.zip','--checkpoint','models/fusdense.pt','--split','test','--exclude-manifest','reports/fer_training_manifest.json')
    Invoke-PythonStage -Stage 'automated_tests' -PythonArguments @('-m','pytest','-q','tests','--junitxml=reports/pytest.xml')
    Invoke-PythonStage -Stage 'api_integration' -PythonArguments @('-m','scripts.verify_end_to_end')
    Invoke-PythonStage -Stage 'measured_reports' -PythonArguments @('-m','scripts.update_research_results')
    Write-Stage 'frontend_build'
    Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
    try{& $runtimeNode node_modules/vite/bin/vite.js build --configLoader native;if($LASTEXITCODE -ne 0){throw 'Frontend build failed'}}finally{Pop-Location}
    Write-Stage 'powerpoint_build_and_validation'
    $oldDeck=Join-Path $projectRoot 'output/RagaMoodSync_Research.pptx'
    $preservedDeck=Join-Path $projectRoot 'reports/baseline_pilot/RagaMoodSync_Research.pptx'
    if(!(Test-Path -LiteralPath $preservedDeck)){Copy-Item -LiteralPath $oldDeck -Destination $preservedDeck}
    & $runtimeNode reports/slides_build/build_deck.mjs
    if($LASTEXITCODE -ne 0){throw 'Presentation build/validation failed'}
    Copy-Item -LiteralPath (Join-Path $projectRoot 'output/RagaMoodSync_Research_Updated.pptx') -Destination $oldDeck
    Write-Stage 'artifacts_built_pending_visual_review'
}catch{
    [System.IO.File]::WriteAllText($statusFile,(@{stage='failed';error=$_.Exception.Message;updated=(Get-Date).ToString('o')}|ConvertTo-Json))
    throw
}
