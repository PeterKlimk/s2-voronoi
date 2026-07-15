<#
.SYNOPSIS
  Build a native bench_voronoi.exe with profile-guided optimization.

.EXAMPLE
  .\scripts\pgo_build.ps1 -Preset balanced
  .\scripts\pgo_build.ps1 -Preset fib

The balanced preset trains Fibonacci, uniform, clustered, and mega paths. The
fib preset maximizes the common well-distributed Fibonacci path, at the cost of
some performance on the adversarial mega distribution.
#>
param(
  [ValidateSet("balanced", "fib")]
  [string]$Preset = "balanced",
  [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$hostTriple = ((rustc -vV | Select-String '^host:').Line -replace '^host:\s*', '').Trim()
$sysroot = (rustc --print sysroot).Trim()
$llvmProfdata = Join-Path $sysroot "lib\rustlib\$hostTriple\bin\llvm-profdata.exe"
if (-not (Test-Path $llvmProfdata)) {
  throw "Missing $llvmProfdata. Install it with: rustup component add llvm-tools-preview"
}

if (-not $OutputRoot) { $OutputRoot = Join-Path $repo "target\pgo" }
$runId = "{0}-{1}" -f (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ"), $PID
$runDir = Join-Path $OutputRoot "$Preset-$runId"
$profileDir = Join-Path $runDir "profiles"
$generateTarget = Join-Path $runDir "generate"
$useTarget = Join-Path $runDir "use"
$outputExe = Join-Path $runDir "bench_voronoi-$Preset.exe"
[void](New-Item -ItemType Directory -Force $profileDir)

$oldRustFlags = $env:RUSTFLAGS
$oldCargoTargetDir = $env:CARGO_TARGET_DIR
$oldProfileFile = $env:LLVM_PROFILE_FILE
$baseRustFlags = "$oldRustFlags -C target-cpu=native -C force-frame-pointers=yes".Trim()

function Invoke-Checked([scriptblock]$Command) {
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

function Train([string[]]$Arguments) {
  Write-Host "Training: bench_voronoi $Arguments" -ForegroundColor Cyan
  & $script:generateExe @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}

try {
  $env:CARGO_TARGET_DIR = $generateTarget
  $env:RUSTFLAGS = "$baseRustFlags -C profile-generate=$profileDir"
  $env:LLVM_PROFILE_FILE = Join-Path $profileDir "default_%m.profraw"

  Write-Host "Building instrumented binary ($Preset)..." -ForegroundColor Cyan
  Invoke-Checked { cargo build --release --features tools --bin bench_voronoi }
  $script:generateExe = Join-Path $generateTarget "release\bench_voronoi.exe"

  Train @("2.5m", "--no-preprocess")
  if ($Preset -eq "balanced") {
    Train @("1m", "--dist", "uniform", "--no-preprocess")
    Train @("500k", "--dist", "clustered", "--no-preprocess")
    Train @("500k", "--dist", "mega", "--no-preprocess")
  }

  $mergedProfile = Join-Path $profileDir "merged.profdata"
  Invoke-Checked { & $llvmProfdata merge -o $mergedProfile $profileDir }

  $env:CARGO_TARGET_DIR = $useTarget
  $env:RUSTFLAGS = "$baseRustFlags -C profile-use=$mergedProfile"
  Remove-Item Env:\LLVM_PROFILE_FILE -ErrorAction SilentlyContinue

  Write-Host "Building profile-optimized binary..." -ForegroundColor Cyan
  Invoke-Checked { cargo build --release --features tools --bin bench_voronoi }
  Copy-Item (Join-Path $useTarget "release\bench_voronoi.exe") $outputExe

  @(
    "preset=$Preset"
    "rustc=$((rustc -Vv) -join ' ')"
    "rustflags=$env:RUSTFLAGS"
    "profile=$mergedProfile"
  ) | Set-Content (Join-Path $runDir "manifest.txt")

  Write-Host "PGO binary: $outputExe" -ForegroundColor Green
} finally {
  $env:RUSTFLAGS = $oldRustFlags
  $env:CARGO_TARGET_DIR = $oldCargoTargetDir
  $env:LLVM_PROFILE_FILE = $oldProfileFile
}
