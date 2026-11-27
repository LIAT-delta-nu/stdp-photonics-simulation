$venv = Join-Path $PSScriptRoot "stdp_sim_venv\Scripts\Activate.ps1"

if (-not (Test-Path $venv)) {
    Write-Error "Virtual environment not found at $venv"
    return
}

. $venv
