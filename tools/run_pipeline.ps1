# Full Phase 1 pipeline: augment -> train -> export -> evaluate.
#
# Every step is resumable, so re-running this after an interruption (reboot,
# closed terminal, sleep) continues where it stopped. Progress goes to
# data\pipeline.log.
#
#   powershell -ExecutionPolicy Bypass -File tools\run_pipeline.ps1
#   Get-Content data\pipeline.log -Wait          # follow it

param(
    [string]$Python = "$env:USERPROFILE\venvs\wisense\Scripts\python.exe",
    [string]$Arch = "small",
    [int]$Epochs = 300,
    [string[]]$TrainExtra = @()
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force data | Out-Null
$log = "data\pipeline.log"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:PYTHONUNBUFFERED = "1"

function Step([string]$name, [string[]]$pyArgs) {
    "`n===== $(Get-Date -Format s)  $name" | Add-Content $log
    # cmd's redirection writes plain bytes; PowerShell 5.1's *>> writes UTF-16.
    cmd /c "`"$Python`" $($pyArgs -join ' ') >> $log 2>&1"
    if ($LASTEXITCODE -ne 0) {
        "===== $(Get-Date -Format s)  FAILED: $name (exit $LASTEXITCODE)" | Add-Content $log
        exit $LASTEXITCODE
    }
}

Step "augment"  @("-m", "wisense.augment", "--subcarriers", "47", "--workers", "11")
Step "train"    (@("-m", "wisense.train", "--arch", $Arch, "--resume", "--epochs", "$Epochs") + $TrainExtra)
Step "export"   @("-m", "wisense.export", "artifacts\$Arch")
Step "evaluate" @("-m", "wisense.evaluate", "artifacts\$Arch")
"`n===== $(Get-Date -Format s)  PIPELINE DONE" | Add-Content $log
