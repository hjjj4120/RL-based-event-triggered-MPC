param(
    [int]$Episodes = 0
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ConfigPath = Join-Path $RepoRoot "configs\paper_grid.json"
$Config = Get-Content $ConfigPath | ConvertFrom-Json
$EpisodeCount = $Config.episodes
if ($Episodes -gt 0) {
    $EpisodeCount = $Episodes
}

Push-Location $RepoRoot
try {
    $ResultsRoot = Join-Path $RepoRoot "results\paper_runs"
    foreach ($run in $Config.runs) {
        foreach ($seed in $Config.seeds) {
            $RunDir = Join-Path $ResultsRoot (Join-Path $run.name ("seed_" + $seed))
            $Args = @(
                "train_ppo.py",
                "--episodes", $EpisodeCount,
                "--seed", $seed,
                "--rho", 0,
                "--run_dir", $RunDir,
                "--sph", $run.sph,
                "--path_set", $run.path_set
            )
            if ($run.PSObject.Properties.Name -contains "np_eco") {
                $Args += @("--np_eco", $run.np_eco)
            }
            if ($run.PSObject.Properties.Name -contains "np_pro") {
                $Args += @("--np_pro", $run.np_pro)
            }
            python @Args
        }
    }

    python (Join-Path $RepoRoot "scripts\make_paper_artifacts.py") --results_root $ResultsRoot --out_dir (Join-Path $RepoRoot "paper_artifacts")
}
finally {
    Pop-Location
}
