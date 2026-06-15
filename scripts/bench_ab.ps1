<#
.SYNOPSIS
  Windows interleaved A/B of bench_voronoi "Total time" between two git refs.

  Builds each ref's bench_voronoi.exe once, then runs them alternating
  (base, cand, base, cand, ...) to cancel slow thermal/clock drift, and reports
  per-ref median + how often the candidate beat the base (sign test).

.EXAMPLE
  # default: main vs the AoS-positions branch, 1m uniform, single-thread, 12 rounds
  .\scripts\bench_ab.ps1

.EXAMPLE
  .\scripts\bench_ab.ps1 -Size 2.5m -Rounds 16 -Dist uniform
  .\scripts\bench_ab.ps1 -Base main -Cand agent/aos-clip-positions -Threads 0   # 0 = all cores (MT)
#>
param(
  [string]$Base    = "main",
  [string]$Cand    = "agent/aos-clip-positions",
  [string]$Size    = "1m",
  [string]$Dist    = "uniform",
  [int]$Rounds     = 12,
  [int]$Threads    = 1            # 1 = single-thread (lowest variance); 0 = leave rayon default (all cores)
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # repo root (scripts/..)
Set-Location $repo

# Refuse to run with a dirty tree (we switch branches to build).
if (git status --porcelain) { throw "Working tree not clean — commit/stash first (this script checks out other refs)." }
$startRef = (git rev-parse --abbrev-ref HEAD).Trim()
if ($startRef -eq "HEAD") { $startRef = (git rev-parse HEAD).Trim() }  # detached

$baseExe = Join-Path $env:TEMP "bench_base.exe"
$candExe = Join-Path $env:TEMP "bench_cand.exe"

function Build-Ref([string]$ref, [string]$outExe) {
  Write-Host "Building $ref ..." -ForegroundColor Cyan
  git checkout --quiet $ref
  cargo build --release --features tools --bin bench_voronoi
  Copy-Item (Join-Path $repo "target\release\bench_voronoi.exe") $outExe -Force
}

try {
  Build-Ref $Base $baseExe
  Build-Ref $Cand $candExe
} finally {
  git checkout --quiet $startRef
}

if ($Threads -gt 0) { $env:RAYON_NUM_THREADS = "$Threads" } else { Remove-Item Env:\RAYON_NUM_THREADS -ErrorAction SilentlyContinue }

function Get-TotalMs([string]$exe) {
  $out = & $exe $Size --dist $Dist 2>&1 | Out-String
  if ($out -match 'Total time:\s*([0-9.]+)\s*ms') { return [double]$Matches[1] }
  throw "Could not parse 'Total time' from $exe`n$out"
}

Write-Host "`nInterleaved A/B: base=$Base  cand=$Cand  size=$Size dist=$Dist threads=$(if($Threads){$Threads}else{'all'}) rounds=$Rounds`n"
$baseT = New-Object System.Collections.Generic.List[double]
$candT = New-Object System.Collections.Generic.List[double]
$candWins = 0
# Warmup (cold start / clock ramp) — discarded.
[void](Get-TotalMs $baseExe); [void](Get-TotalMs $candExe)
for ($i = 1; $i -le $Rounds; $i++) {
  $b = Get-TotalMs $baseExe
  $c = Get-TotalMs $candExe
  $baseT.Add($b); $candT.Add($c)
  $win = if ($c -lt $b) { $candWins++; "cand" } else { "base" }
  "{0,3}  base={1,9:N1}ms  cand={2,9:N1}ms  {3,8:P1}  ({4})" -f $i, $b, $c, (($c - $b) / $b), $win | Write-Host
}

function Median([System.Collections.Generic.List[double]]$xs) {
  $s = $xs | Sort-Object; $n = $s.Count
  if ($n % 2) { return $s[[int][math]::Floor($n/2)] }
  return ($s[$n/2 - 1] + $s[$n/2]) / 2
}
$mb = Median $baseT; $mc = Median $candT
Write-Host ""
Write-Host ("base  median: {0,9:N1} ms" -f $mb)
Write-Host ("cand  median: {0,9:N1} ms   ({0:N1} = {1:P1} vs base)" -f $mc, (($mc - $mb) / $mb))
Write-Host ("cand faster in {0}/{1} rounds" -f $candWins, $Rounds)
