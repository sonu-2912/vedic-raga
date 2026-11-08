$taskRoot='C:\Users\bindu\OneDrive - iitr.ac.in\VEDIC RAGA'
$run=Join-Path $taskRoot 'reports\pure_repair_20261108'
$deadline=(Get-Date).AddMinutes(120)
while(-not (Test-Path -LiteralPath (Join-Path $run 'completion.json'))){
 if((Get-Date) -gt $deadline){throw 'Coverage boundary did not arrive; no completion claimed'}
 $historyPath=Join-Path $run 'training_history.json'
 if(Test-Path -LiteralPath $historyPath){
  $history=Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
  $recent=@($history | Where-Object {$_.phase -eq 'full_network_pure_coverage_repair'})
  if($recent.Count -ge 6){
   if(Test-Path -LiteralPath (Join-Path $run 'best.pt')){Write-Output 'A qualifying checkpoint exists. Keep original declared training budget.';exit 0}
   $owned=Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($taskRoot) -and $_.CommandLine -match '\-m scripts.train_pure_coverage_repair' -and $_.CommandLine -match '\-\-run-dir reports/pure_repair_20261108' }
   foreach($proc in $owned){Stop-Process -Id $proc.ProcessId -ErrorAction SilentlyContinue}
   & "$taskRoot\.venv\Scripts\python.exe" -m scripts.record_coverage_transition
   if($LASTEXITCODE -ne 0){throw 'Completed-epoch transition recording failed'}
   Write-Output 'Saved completed coverage epochs; conditional orientation transition recorded.'
   exit 0
  }
 }
 Start-Sleep -Seconds 5
}
Write-Output 'Coverage completed naturally; no intervention.'
