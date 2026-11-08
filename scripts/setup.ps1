$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 from python.org, then rerun setup.' }
}
& .venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Push-Location frontend
try { npm ci --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }; npm run build; if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' } }
finally { Pop-Location }
Write-Output 'Setup complete. Run .\scripts\start-backend.ps1 and open http://127.0.0.1:8000'
