$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

New-Item -ItemType Directory -Force -Path "results" | Out-Null

$buildDir = "build"
cmake -S . -B $buildDir
cmake --build $buildDir --config Release

$exe = Join-Path $buildDir "Release\rotate_part_a.exe"
if (-not (Test-Path $exe)) {
    $exe = Join-Path $buildDir "rotate_part_a.exe"
}
if (-not (Test-Path $exe)) {
    Write-Error "rotate_part_a.exe not found"
}

Write-Host "Running $exe ..."
& $exe 2>&1 | Tee-Object -FilePath "results\rotate_benchmark.txt"
Write-Host "Done. Log: results/rotate_benchmark.txt"
