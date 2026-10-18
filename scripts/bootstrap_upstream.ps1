# Windows equivalent of bootstrap_upstream.sh. Run from repo root in PowerShell.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path "upstream\.git") {
    git -C upstream pull --ff-only
} else {
    git clone https://github.com/metrica-sports/more-than-a-hack-2027 upstream
}

# Junction so the upstream starter sees our data/ folder (no admin needed for junctions).
if (-not (Test-Path "upstream\data")) {
    cmd /c mklink /J "upstream\data" "$(Resolve-Path data)"
}

pip install -r upstream\requirements.txt
Write-Host "upstream ready. Try: python upstream\evaluation\run_evaluation.py"
