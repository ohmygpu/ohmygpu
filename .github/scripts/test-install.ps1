# CI test for install.ps1 (see .github/workflows/installers.yml): a real install into a
# temp dir against the published releases - piped like `irm | iex`, then pin an older
# release, upgrade in place through the PATH lookup, and finally "nothing to do".
# Run with Windows PowerShell 5.1 and PowerShell 7:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .github\scripts\test-install.ps1
$ErrorActionPreference = 'Stop'

$dir = Join-Path $env:RUNNER_TEMP 'omg-bin'
function Version-Of([string]$exe) { ((& $exe --version) -split '\s+')[1] }
function Step([string]$name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Step "fresh install into $dir (piped, like irm | iex)"
$env:OHMYGPU_INSTALL_DIR = $dir
Get-Content -Raw install.ps1 | Invoke-Expression
foreach ($exe in 'ohmygpu.exe', 'ohmygpu-runtime.exe', 'omg.exe') {
    $v = Version-Of (Join-Path $dir $exe)
    if (-not $v) { throw "$exe does not run" }
    Write-Host "$exe -> $v"
}
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*omg-bin*") { throw "install dir was not added to the user PATH: $userPath" }
Remove-Item Env:OHMYGPU_INSTALL_DIR

Step "pin v0.4.0 into the same dir"
.\install.ps1 -InstallDir $dir -Version v0.4.0 -NoModifyPath
if ((Version-Of (Join-Path $dir 'ohmygpu.exe')) -ne '0.4.0') { throw "pinning v0.4.0 failed" }

Step "upgrade in place: no -InstallDir, the existing install is found on PATH"
$env:Path = "$dir;$env:Path"
.\install.ps1 -NoModifyPath
if ((Version-Of (Join-Path $dir 'ohmygpu.exe')) -eq '0.4.0') { throw "upgrade in place failed" }
if ((Version-Of (Join-Path $dir 'ohmygpu-runtime.exe')) -eq '0.4.0') { throw "runtime was not upgraded" }

Step "same version again: nothing to do"
$out = .\install.ps1 -NoModifyPath 6>&1 | Out-String
Write-Host $out
if ($out -notmatch 'nothing to do') { throw "expected 'nothing to do', got: $out" }

Write-Host "`ninstall.ps1: all checks passed ($($PSVersionTable.PSVersion))" -ForegroundColor Green
