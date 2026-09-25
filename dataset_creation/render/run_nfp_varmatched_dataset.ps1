# Generate the variance-matched NFP ball dataset using Kubric Docker.
#
# Variance-matched = same renderer/scene as v1/v2/v3, but the velocity profiles
# follow the variance-matched design (dataset_creation/design_variance_matched.py):
# the five temporal concepts are reweighted to have as-equal-as-possible within-video
# variance (acceleration boosted from ~0.03 to ~0.22 via the oscillating-speed family)
# while all ten pairwise couplings stay ~0. Start positions remain uniform and
# independent of the profile (S1/S3), so the NFP proof holds exactly as in v1/v2/v3.
#
# Prerequisite: dataset_creation\specs\nfp_varmatched_profile_spec.json
#   (regenerate with:  python -m dataset_creation.design_variance_matched --N 3000)
#
# Usage (from PowerShell, in dataset_creation\render\):
#   .\run_nfp_varmatched_dataset.ps1                 # all 3000 videos
#   .\run_nfp_varmatched_dataset.ps1 -NVideos 2      # smoke test
#   .\run_nfp_varmatched_dataset.ps1 -Start 0 -End 1499   # shard
#
# Output: .\output\nfp_varmatched\v00000\ ... v02999\

param(
    [int]$NVideos = 3000,
    [int]$Start   = 0,
    [int]$End     = -1,
    [int]$Seed    = 0
)

# Mount dataset_creation\ as /kubric_scripts so both the renderer
# (nfp_ball_dataset.py) and the spec (specs\...) are visible in the container.
$ScriptDir  = $PSScriptRoot                         # dataset_creation\render\
$PkgDir     = Split-Path $ScriptDir -Parent         # dataset_creation\
$OutDir     = Join-Path $ScriptDir "output\nfp_varmatched"
$Image      = "kubricdockerhub/kubruntu"
$SpecRel    = "specs/nfp_varmatched_profile_spec.json"   # forward slashes for the container

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($End -lt 0) { $End = $NVideos - 1 }

if (-not (Test-Path (Join-Path $PkgDir "specs\nfp_varmatched_profile_spec.json"))) {
    Write-Error "Missing dataset_creation\specs\nfp_varmatched_profile_spec.json (regenerate with 'python -m dataset_creation.design_variance_matched')"
    exit 1
}

Write-Host "=== NFP variance-matched dataset: videos $Start .. $End (seed=$Seed) ===" -ForegroundColor Cyan
Write-Host "Output: $OutDir"

docker run --rm `
    --gpus all `
    --volume "${PkgDir}:/kubric_scripts:ro" `
    --volume "${OutDir}:/output" `
    $Image `
    /usr/bin/python3 /kubric_scripts/nfp_ball_dataset.py `
        --output_dir /output `
        --n_videos   $NVideos `
        --start_idx  $Start `
        --end_idx    $End `
        --seed       $Seed `
        --profile_spec /kubric_scripts/$SpecRel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker run failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== Done. Output in $OutDir ===" -ForegroundColor Green
