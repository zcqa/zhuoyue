param([ValidateSet('menu','offline','realtime','microphone','experiments','devices','tests')][string]$Mode = 'menu')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run .\setup.ps1 first.' }
if ($Mode -eq 'menu') {
    Write-Host '1. Local file transcription + speaker labels (sample C)'
    Write-Host '2. Realtime paced replay (sample E, about 2 minutes)'
    Write-Host '3. Live microphone transcription (120 seconds)'
    Write-Host '4. Run experiments and generate report'
    Write-Host '5. List audio devices'
    Write-Host '6. Run regression tests'
    $choice = Read-Host 'Choose 1-6'
    $Mode = switch ($choice) { '1' {'offline'} '2' {'realtime'} '3' {'microphone'} '4' {'experiments'} '5' {'devices'} '6' {'tests'} default { throw 'Invalid choice' } }
}
switch ($Mode) {
    'offline' { & $python -m meeting_asr offline data\samples\C.wav --diarize --output outputs\manual_offline }
    'realtime' { & $python -m meeting_asr realtime data\samples\E.wav --duration 0 --output outputs\manual_realtime }
    'microphone' { & $python -m meeting_asr realtime --mic --duration 120 --output outputs\manual_microphone }
    'experiments' { & $python scripts\run_experiments.py --device cpu; if ($LASTEXITCODE -eq 0) { & $python scripts\build_report.py } }
    'devices' { & $python -m meeting_asr devices }
    'tests' { & $python -m pytest -q }
}
if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
