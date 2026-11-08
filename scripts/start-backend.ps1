$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runningHealth=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3
    if($runningHealth.status -eq 'ok' -and $runningHealth.ragas -eq 72 -and $runningHealth.models.fusdense){
        Write-Host 'RagaMoodSync is already running: http://127.0.0.1:8000/'
        return
    }
} catch {
    # No matching healthy service; uvicorn will report any unrelated port conflict.
}
& .venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
