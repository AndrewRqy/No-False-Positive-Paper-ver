# Generate the NFP v2 (decorrelated) ball dataset using Kubric Docker.
#
# v2 = same renderer/scene as v1, but velocity profiles follow the joint
# decorrelated design (analysis/design_decorrelated_stimulus.py --joint):
# all 10 pairwise within-video covariances among {speed, vel_x, vel_y,
# accel_mag, direction} are zero at the dataset level. Start positions remain
# uniform and independent of the profile (S1/S3), so the NFP proof holds.
#
# Prerequisite: data\nfp_v2_profile_spec.json (copied from
# local_runs/steering/nfp_v2_profile_spec.json).
#
# Usage (from PowerShell):
#   .\run_nfp_v2_dataset.ps1               # all 3000 videos
#   .\run_nfp_v2_dataset.ps1 -NVideos 2    # smoke test
#   .\run_nfp_v2_dataset.ps1 -Start 0 -End 1499   # shard
#
# Output: .\output\nfp_v2\v00000\ ... v02999\

param(
    [int]$NVideos = 3000,
    [int]$Start   = 0,
    [int]$End     = -1,
    [int]$Seed    = 0
)

$ScriptDir  = $PSScriptRoot
$OutDir     = Join-Path $ScriptDir "output\nfp_v2"
$Image      = "kubricdockerhub/kubruntu"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($End -lt 0) { $End = $NVideos - 1 }

if (-not (Test-Path (Join-Path $ScriptDir "nfp_v2_profile_spec.json"))) {
    Write-Error "Missing data\nfp_v2_profile_spec.json (copy from local_runs/steering/)"
    exit 1
}

Write-Host "=== NFP v2 (decorrelated) dataset: videos $Start .. $End (seed=$Seed) ===" -ForegroundColor Cyan
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
        --profile_spec /kubric_scripts/nfp_v2_profile_spec.json

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
