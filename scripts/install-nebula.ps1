# Builds nebula.exe and nebula-daemon.exe (release) and copies them to F:\Nebula\bin\.
# Stop the daemon first (`nebula daemon stop`); a running nebula-daemon.exe can't be replaced.
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$bin = 'F:\Nebula\bin'
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = 'F:\Nebula\build\target' }

$running = Get-Process -Name 'nebula-daemon' -ErrorAction SilentlyContinue
if ($running) { throw "nebula-daemon is running (pid $($running.Id -join ', ')); run 'nebula daemon stop' first" }

Push-Location $repo
try {
    cargo build --release --locked -p nebula-cli -p nebula-daemon
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

New-Item -ItemType Directory -Force -Path $bin | Out-Null
foreach ($exe in 'nebula.exe', 'nebula-daemon.exe') {
    Copy-Item -Force (Join-Path $env:CARGO_TARGET_DIR "release\$exe") (Join-Path $bin $exe)
}
$tools = 'F:\Nebula\tools'
New-Item -ItemType Directory -Force -Path $tools | Out-Null
$targetDevtools = Join-Path $repo 'tools\nebula-devtools'
$destDevtools = Join-Path $tools 'nebula-devtools'
if (-not (Test-Path $destDevtools)) {
    New-Item -ItemType Junction -Path $destDevtools -Target $targetDevtools | Out-Null
}

$hydrogenBin = Join-Path $repo 'hydrogen\build\bin\hydrogen_engine.exe'
if (Test-Path $hydrogenBin) {
    $runtimeHydrogen = 'F:\Nebula\runtime\hydrogen'
    New-Item -ItemType Directory -Force -Path $runtimeHydrogen | Out-Null
    Copy-Item -Force $hydrogenBin (Join-Path $runtimeHydrogen 'hydrogen_engine.exe')
    Write-Host "Installed hydrogen_engine to $runtimeHydrogen"
}

Write-Host "Installed to $bin"
& (Join-Path $bin 'nebula.exe') --version
