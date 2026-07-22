$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run .\scripts\setup.ps1 first."
}
Set-Location $ProjectRoot
$env:RAG_UPDATE_SUPERVISED = "1"
$UpdateRestartCode = 75
$UpdateMarker = Join-Path $ProjectRoot ".reference-desk-update.json"

while ($true) {
    & $Python main.py serve
    $ServerExitCode = $LASTEXITCODE
    if ($ServerExitCode -ne $UpdateRestartCode) {
        exit $ServerExitCode
    }

    Write-Host "Applying the Reference Desk update..." -ForegroundColor Cyan
    $DependencyRefreshFailed = $false
    if (Test-Path $UpdateMarker) {
        try {
            $UpdateState = Get-Content -LiteralPath $UpdateMarker -Raw | ConvertFrom-Json
            if ($UpdateState.dependencies_changed) {
                $ProfilePath = Join-Path $ProjectRoot ".rag-profile"
                $RagBackend = if (Test-Path $ProfilePath) { (Get-Content -LiteralPath $ProfilePath -Raw).Trim() } else { "cpu" }
                $BackendRequirements = switch ($RagBackend) {
                    "cuda" { "requirements-cuda.txt" }
                    "rocm" { "requirements-rocm-windows.txt" }
                    default { "requirements-cpu.txt" }
                }
                & $Python -m pip install -r $BackendRequirements
                if ($LASTEXITCODE -ne 0) { $DependencyRefreshFailed = $true }
                if (-not $DependencyRefreshFailed) {
                    & $Python -m pip install -r requirements-base.txt
                    if ($LASTEXITCODE -ne 0) { $DependencyRefreshFailed = $true }
                }
            }
        } catch {
            $DependencyRefreshFailed = $true
            Write-Warning "Could not read the update state: $($_.Exception.Message)"
        }
        Remove-Item -LiteralPath $UpdateMarker -Force -ErrorAction SilentlyContinue
    }
    if ($DependencyRefreshFailed) {
        Write-Warning "Some dependencies could not be refreshed. The app will still restart; run SETUP.bat if it reports a missing package."
    }
    Write-Host "Restarting Reference Desk..." -ForegroundColor Green
}
