[CmdletBinding()]
param(
    [string]$Repository = "reference-desk",
    [ValidateSet("private", "public")]
    [string]$Visibility = "private"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "Install GitHub CLI from https://cli.github.com/ and run 'gh auth login'."
}
if (-not (Test-Path ".git")) { git init }
$GitName = git config user.name
$GitEmail = git config user.email
if (-not $GitName -or -not $GitEmail) {
    throw "Configure Git first with 'git config --global user.name YOUR-NAME' and 'git config --global user.email YOUR-EMAIL'."
}
& .\.venv\Scripts\python.exe scripts\export_release.py --check *> $null
if ($LASTEXITCODE -ne 0) { throw "The private-data export check failed." }
git add .
git commit -m "Prepare portable Reference Desk release"
gh repo create $Repository --source . --remote origin --push "--$Visibility"
Write-Host "Published $Repository as $Visibility." -ForegroundColor Green
