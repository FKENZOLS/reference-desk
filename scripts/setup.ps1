[CmdletBinding()]
param(
    [ValidateSet("auto", "cuda", "rocm", "cpu")]
    [string]$Backend = "auto",
    [string]$Python = "python",
    [switch]$SkipOllamaPull
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Resolve-Backend([string]$Requested) {
    if ($Requested -ne "auto") { return $Requested }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return "cuda" }
    try {
        $DisplayNames = Get-CimInstance Win32_VideoController |
            ForEach-Object { $_.Name } |
            Where-Object { $_ }
        if (($DisplayNames -join " ") -match "AMD|Radeon") { return "rocm" }
    } catch {
        Write-Host "Could not inspect the display adapter; selecting CPU." -ForegroundColor Yellow
    }
    return "cpu"
}

$Selected = Resolve-Backend $Backend
$PythonDetails = & $Python -c "import platform, struct, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.python_implementation()}|{struct.calcsize('P') * 8}')"
if ($LASTEXITCODE -ne 0) { throw "Python could not be started with '$Python'." }
$PythonVersion, $PythonImplementation, $PythonBits = $PythonDetails.Trim().Split("|")
if ($PythonVersion -ne "3.12" -or $PythonImplementation -ne "CPython" -or $PythonBits -ne "64") {
    throw "Reference Desk requires 64-bit CPython 3.12. Found $PythonVersion $PythonImplementation $PythonBits-bit."
}

Write-Host "Preparing Reference Desk for $Selected" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
}
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip wheel

switch ($Selected) {
    "cuda" { & $VenvPython -m pip install -r requirements-cuda.txt }
    "rocm" { & $VenvPython -m pip install -r requirements-rocm-windows.txt }
    "cpu"  { & $VenvPython -m pip install -r requirements-cpu.txt }
}
if ($LASTEXITCODE -ne 0) { throw "The $Selected PyTorch installation failed." }

& $VenvPython -m pip install -r requirements-base.txt
if ($LASTEXITCODE -ne 0) { throw "The application dependency installation failed." }
Set-Content -LiteralPath ".rag-profile" -Value $Selected -Encoding ascii

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    if (-not $SkipOllamaPull) { & ollama pull embeddinggemma }
} else {
    Write-Host "Ollama was not found. Install it, then run: ollama pull embeddinggemma" -ForegroundColor Yellow
}

& $VenvPython scripts\doctor.py --expect $Selected
if ($LASTEXITCODE -ne 0) {
    throw "The compatibility check found a blocking problem. Read the messages above."
}
Write-Host "Ready. Start with .\start.ps1" -ForegroundColor Green
