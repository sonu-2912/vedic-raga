$ErrorActionPreference='Stop'
$projectRoot=(Get-Location).Path
$historyFile=Join-Path $projectRoot 'reports/pure_fusdense_20261108/training_history.json'
$completionFile=Join-Path $projectRoot 'reports/pure_fusdense_20261108/completion.json'
while ($true) {
 if (Test-Path -LiteralPath $completionFile) { break }
 if (Test-Path -LiteralPath $historyFile) {
  $history=Get-Content -LiteralPath $historyFile -Raw | ConvertFrom-Json
  $completed=@($history | Where-Object phase -eq 'full_network_pure_supcon_rdrop').Count
  if ($completed -ge 6) { break }
 }
 Start-Sleep -Seconds 10
}
if (-not (Test-Path -LiteralPath $completionFile)) {
 $targets=Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*-m scripts.train_pure_fusdense --run-dir reports/pure_fusdense_20261108*' -and $_.CommandLine -like '*OneDrive - iitr.ac.in*' }
 foreach($target in $targets) { Stop-Process -Id $target.ProcessId -ErrorAction SilentlyContinue }
}
$pythonPath=Join-Path $projectRoot '.venv/Scripts/python.exe'
& $pythonPath -m scripts.record_pure_budget_stop
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonPath -u -m scripts.train_pure_native_residual --run-dir reports/pure_native_residual_20261108 --threads 4 *> (Join-Path $projectRoot 'reports/pure_native_residual_20261108/training.log')
exit $LASTEXITCODE
