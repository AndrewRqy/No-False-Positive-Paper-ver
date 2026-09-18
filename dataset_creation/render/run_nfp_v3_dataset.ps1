# Generate an NFP v3 (targeted-correlation) ball dataset using Kubric Docker.
#
# v3 = same renderer/scene as v1/v2, but the velocity profiles follow the
# targeted-correlation design (analysis/design_correlated_stimulus.py): one
# temporal pair is held at within-video correlation ~0.9 while the others stay
# ~0. Two variants:
#   S : corr(accel_mag, speed) = 0.9   -> nfp_v3_S_profile_spec.json
#   X : corr(accel_mag, vel_x) = 0.9   -> nfp_v3_X_profile_spec.json
# Start positions remain uniform and independent of the profile (S1/S3), so the
# NFP proof holds exactly as in v1/v2.
#
# Prerequisite: data\nfp_v3_<Variant>_profile_spec.json (copied from
# local_runs/steering/).
#
# Usage (from PowerShell):
#   .\run_nfp_v3_dataset.ps1 -Variant S              # all 3000 videos
#   .\run_nfp_v3_dataset.ps1 -Variant S -NVideos 2   # smoke test
#   .\run_nfp_v3_dataset.ps1 -Variant X -Start 0 -End 1499   # shard
#
# Output: .\output\nfp_v3_<Variant>\v00000\ ... v02999\

param(
    [Parameter(Mandatory=$true)][ValidateSet("S","X")][string]$Variant,
    [int]$NVideos = 3000,
    [int]$Start   = 0,
    [int]$End     = -1,
    [int]$Seed    = 0
)

$ScriptDir  = $PSScriptRoot
$OutDir     = Join-Path $ScriptDir "output\nfp_v3_$Variant"
$Image      = "kubricdockerhub/kubruntu"
$SpecName   = "nfp_v3_${Variant}_profile_spec.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($End -lt 0) { $End = $NVideos - 1 }

if (-not (Test-Path (Join-Path $ScriptDir $SpecName))) {
    Write-Error "Missing data\$SpecName (copy from local_runs/steering/)"
    exit 1
}

Write-Host "=== NFP v3 variant $Variant dataset: videos $Start .. $End (seed=$Seed) ===" -ForegroundColor Cyan
Write-Host "Output: $OutDir"

docker run --rm `
    --gpus all `
    --volume "${ScriptDir}:/kubric_scripts:ro" `
    --volume "${OutDir}:/output" `
    $Image `
    /usr/bin/python3 /kubric_scripts/nfp_ball_dataset.py `
        --output_dir /output `
        --n_videos   $NVideos `
        --start_idx  $Start `
        --end_idx    $End `
        --seed       $Seed `
        --profile_spec /kubric_scripts/$SpecName

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
