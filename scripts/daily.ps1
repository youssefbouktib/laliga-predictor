# Daily update: fresh data -> predictions -> paper-trade log -> settle finished bets.
# Register it once with Windows Task Scheduler (runs every day at 09:00):
#   schtasks /Create /SC DAILY /ST 09:00 /TN "LaLigaPredictor" `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\path\to\laliga-predictor\scripts\daily.ps1\""
#
# Not "Stop": in Windows PowerShell 5.1 every stderr line of a native program
# (Python warnings, tracebacks) becomes an error record, and "Stop" would kill
# the script on the first one before anything is logged. Failures are detected
# with $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Set-Location $root
New-Item -ItemType Directory -Force (Join-Path $root "reports") | Out-Null
$log = Join-Path $root "reports\daily.log"
$env:PYTHONIOENCODING = "utf-8"   # team names with accents survive the pipe
$env:PYTHONUNBUFFERED = "1"       # keep stdout and stderr lines in order
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Log($text) { $text | Out-File $log -Append -Encoding utf8 }

Write-Log "===== $(Get-Date -Format s) ====="
foreach ($cmd in @("download", "predict", "settle")) {
    # 2>&1 sends stderr into the log too; "$_" turns error records into plain text
    & $py run.py $cmd 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Log "run.py $cmd failed with exit code $code"
        exit 1
    }
}
Write-Log "===== done $(Get-Date -Format s) ====="
