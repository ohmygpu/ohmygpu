#Requires -Version 5.1
<#
.SYNOPSIS
    OhMyGPU Runtime installer for Windows (x64) - https://github.com/ohmygpu/ohmygpu

.DESCRIPTION
    Installs the latest GitHub release of ohmygpu-runtime.exe (the runtime) and
    ohmygpu.exe (the CLI, plus an omg.exe copy), verifies the SHA-256 checksum
    shipped with the release, adds the install directory to your user PATH, and
    upgrades an existing install in place. Once installed, `omg upgrade` does the
    same from the CLI.

        irm https://raw.githubusercontent.com/ohmygpu/ohmygpu/main/install.ps1 | iex

    Options - environment variables for the one-liner, parameters when running the file:

        OHMYGPU_VERSION=v0.5.0      -Version v0.5.0     release to install (default: latest)
        OHMYGPU_INSTALL_DIR=DIR     -InstallDir DIR     install directory (default: the existing
                                                        install dir, else %LOCALAPPDATA%\Programs\ohmygpu)
        OHMYGPU_NO_MODIFY_PATH=1    -NoModifyPath       do not touch the user PATH
        OHMYGPU_FORCE=1             -Force              reinstall even if that version is installed

    Running the downloaded file: powershell -ExecutionPolicy Bypass -File install.ps1 [-Version v0.5.0]
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '', Justification = 'installer console output')]
[CmdletBinding()]
param(
    [string]$Version = $env:OHMYGPU_VERSION,
    [string]$InstallDir = $env:OHMYGPU_INSTALL_DIR,
    [switch]$NoModifyPath,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar on Windows PowerShell 5.1

$Repo = 'ohmygpu/ohmygpu'
$Releases = "https://github.com/$Repo/releases"
$Target = 'x86_64-pc-windows-msvc'
$Asset = "ohmygpu-$Target.zip"
$Binaries = @('ohmygpu-runtime.exe', 'ohmygpu.exe')

if ($env:OHMYGPU_NO_MODIFY_PATH -eq '1') { $NoModifyPath = $true }
if ($env:OHMYGPU_FORCE -eq '1') { $Force = $true }

function Write-Info([string]$Message) { Write-Host $Message }
function Write-Warn([string]$Message) { Write-Host "warning: $Message" -ForegroundColor Yellow }

# Older Windows PowerShell defaults may not offer TLS 1.2, which GitHub requires.
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { Write-Verbose "could not enable TLS 1.2: $_" }

# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------

$arch = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }   # 32-bit PowerShell on 64-bit Windows
switch ($arch) {
    'AMD64' { }
    'ARM64' { Write-Warn "no native Windows arm64 build yet; installing the x64 build (runs under emulation)" }
    default { throw "unsupported architecture '$arch' (prebuilt Windows binaries exist for x64 only)" }
}

# ---------------------------------------------------------------------------
# which release
# ---------------------------------------------------------------------------

$Tag = $null
if ($Version) {
    $Tag = if ($Version.StartsWith('v')) { $Version } else { "v$Version" }
} else {
    try {
        $latest = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest" `
            -Headers @{ 'User-Agent' = 'ohmygpu-install' } -TimeoutSec 20
        $Tag = $latest.tag_name
    } catch {
        # Rate-limited or offline API: the version-less `latest/download` URLs still work.
        $Tag = $null
    }
}
$Base = if ($Tag) { "$Releases/download/$Tag" } else { "$Releases/latest/download" }

# ---------------------------------------------------------------------------
# existing install (upgrade in place) and install directory
# ---------------------------------------------------------------------------

$existing = Get-Command 'ohmygpu.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
$existingVersion = $null
$existingDir = $null
if ($existing) {
    $existingDir = Split-Path -Parent $existing.Path
    try { $existingVersion = ((& $existing.Path --version) -split '\s+')[1] } catch { Write-Verbose "existing ohmygpu.exe did not report a version: $_" }
}

if (-not $InstallDir) {
    if ($existingDir) {
        $InstallDir = $existingDir
    } else {
        $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\ohmygpu'
    }
}
$InstallDir = [IO.Path]::GetFullPath($InstallDir)

if (-not $Force -and $Tag -and $existingVersion -and ("v$existingVersion" -eq $Tag) -and ($existingDir -eq $InstallDir)) {
    Write-Info "OhMyGPU Runtime $Tag is already installed in $InstallDir - nothing to do (OHMYGPU_FORCE=1 / -Force reinstalls)."
    return
}

# ---------------------------------------------------------------------------
# download, verify, install
# ---------------------------------------------------------------------------

$Tmp = Join-Path ([IO.Path]::GetTempPath()) ('ohmygpu-install-' + [IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $Tmp -Force | Out-Null
try {
    $zip = Join-Path $Tmp $Asset
    $shown = if ($Tag) { $Tag } else { 'latest' }
    Write-Info "Downloading $Asset ($shown) ..."
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$Base/$Asset" -OutFile $zip
    } catch {
        throw "download failed: $Base/$Asset ($($_.Exception.Message)) - no such release, or no Windows build in it? See $Releases"
    }
    $sumsFile = Join-Path $Tmp 'SHA256SUMS.txt'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$Base/SHA256SUMS.txt" -OutFile $sumsFile
    } catch {
        throw "download failed: $Base/SHA256SUMS.txt ($($_.Exception.Message))"
    }
    $expected = $null
    foreach ($line in Get-Content $sumsFile) {
        $parts = $line.Trim() -split '\s+'
        if ($parts.Count -ge 2 -and $parts[1].TrimStart('*') -eq $Asset) { $expected = $parts[0].ToLower(); break }
    }
    if (-not $expected) { throw "no checksum for $Asset in SHA256SUMS.txt" }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLower()
    if ($actual -ne $expected) { throw "checksum mismatch for $Asset (expected $expected, got $actual) - refusing to install" }

    $extract = Join-Path $Tmp 'extracted'
    Expand-Archive -Path $zip -DestinationPath $extract -Force
    $srcDir = Join-Path $extract "ohmygpu-$Target"
    foreach ($f in $Binaries) {
        if (-not (Test-Path (Join-Path $srcDir $f))) { throw "unexpected archive layout in $Asset (missing $f)" }
    }
    Get-ChildItem -Path $srcDir -Filter '*.exe' | Unblock-File -ErrorAction SilentlyContinue

    # A running .exe cannot be overwritten, but it can be renamed away; leftovers are
    # removed on the next run (and by `omg upgrade`).
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    foreach ($f in $Binaries + @('omg.exe')) {
        $dest = Join-Path $InstallDir $f
        $src = if ($f -eq 'omg.exe') { Join-Path $srcDir 'ohmygpu.exe' } else { Join-Path $srcDir $f }
        $old = "$dest.old"
        if (Test-Path $old) { Remove-Item $old -Force -ErrorAction SilentlyContinue }
        if (Test-Path $dest) { Move-Item $dest $old -Force }
        Copy-Item $src $dest -Force
        if (Test-Path $old) { Remove-Item $old -Force -ErrorAction SilentlyContinue }
    }
} finally {
    Remove-Item $Tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$newVersion = $null
try { $newVersion = ((& (Join-Path $InstallDir 'ohmygpu.exe') --version) -split '\s+')[1] } catch { Write-Verbose "new ohmygpu.exe did not run: $_" }
if (-not $newVersion) { throw "$InstallDir\ohmygpu.exe was installed but does not run" }

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------

function Test-OnPath([string]$PathList, [string]$Dir) {
    foreach ($p in ($PathList -split ';')) {
        if ($p -and ($p.TrimEnd('\') -ieq $Dir.TrimEnd('\'))) { return $true }
    }
    return $false
}

$pathChanged = $false
if (-not $NoModifyPath) {
    # Go through the registry directly so %VAR% entries in the user PATH are kept unexpanded.
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    try {
        $userPath = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if (-not (Test-OnPath $userPath $InstallDir)) {
            $newPath = if ($userPath) { $userPath.TrimEnd(';') + ';' + $InstallDir } else { $InstallDir }
            $key.SetValue('Path', $newPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
            $pathChanged = $true
        }
    } finally {
        $key.Close()
    }
    if ($pathChanged) {
        # Tell running shells/Explorer that the environment changed, so new terminals see it.
        try {
            if (-not ('OhMyGPU.Native' -as [type])) {
                Add-Type -Namespace OhMyGPU -Name Native -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Auto)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
'@
            }
            $result = [UIntPtr]::Zero
            [void][OhMyGPU.Native]::SendMessageTimeout([IntPtr]0xffff, 0x001A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$result)
        } catch { Write-Verbose "could not broadcast the PATH change: $_" }
    }
    if (-not (Test-OnPath $env:Path $InstallDir)) { $env:Path = "$InstallDir;$env:Path" }
}

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

Write-Info ''
if ($existingVersion -and ($existingDir -eq $InstallDir)) {
    Write-Info "Upgraded OhMyGPU Runtime v$existingVersion -> v$newVersion in $InstallDir"
} else {
    Write-Info "Installed OhMyGPU Runtime v$newVersion to $InstallDir (ohmygpu-runtime.exe, ohmygpu.exe, omg.exe)"
}
if ($pathChanged) {
    Write-Info "Added $InstallDir to your user PATH - open a new terminal for it to take effect."
} elseif ($NoModifyPath -and -not (Test-OnPath $env:Path $InstallDir)) {
    Write-Info "$InstallDir is not on your PATH (not modified because of -NoModifyPath / OHMYGPU_NO_MODIFY_PATH)."
}

# A runtime that is already running keeps the old version until restarted.
$port = if ($env:OHMYGPU_PORT) { $env:OHMYGPU_PORT } else { '10692' }
try {
    $health = Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:$port/ohmygpu/v1/health" -TimeoutSec 2
    if ($health.status -eq 'ok') {
        Write-Info ''
        Write-Info "A runtime (v$($health.version)) is running on port $port; restart it to use v${newVersion}:  omg shutdown; omg serve"
    }
} catch { Write-Verbose "no runtime answering on port ${port}: $_" }

Write-Info ''
Write-Info 'Next:'
Write-Info '  omg serve                               # start the runtime (foreground)'
Write-Info '  omg model pull qwen2.5-0.5b-instruct    # download a model (omg model catalog lists more)'
Write-Info '  omg run qwen2.5-0.5b-instruct           # start it, then POST http://127.0.0.1:10692/v1/responses'
Write-Info ''
Write-Info 'Upgrade later with:  omg upgrade   (or run this script again)'
