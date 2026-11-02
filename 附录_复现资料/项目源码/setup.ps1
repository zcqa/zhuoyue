param([string]$IndexUrl = 'https://pypi.tuna.tsinghua.edu.cn/simple')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
}
& '.\.venv\Scripts\python.exe' -m pip install --index-url $IndexUrl -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
& '.\.venv\Scripts\python.exe' scripts\prepare_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
& '.\.venv\Scripts\python.exe' scripts\collect_samples.py
if ($LASTEXITCODE -ne 0) { throw 'Test audio download failed' }
& '.\.venv\Scripts\python.exe' scripts\collect_natural_meeting.py
if ($LASTEXITCODE -ne 0) { throw 'Natural meeting sample download failed' }
Write-Host 'Ready. Run .\start.ps1 or read README.md.'
