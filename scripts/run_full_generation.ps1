$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot
Set-Location $RepoRoot

$LogPath = Join-Path $RepoRoot "output.log"
$StartDate = "20090101"
$EndDate = "20260424"

Set-Content -Path $LogPath -Value ""

function Write-Log {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $Line | Tee-Object -FilePath $LogPath -Append
}

function Quote-CmdArg {
    param([string]$Value)
    if ($Value -notmatch '[\s"&|<>^]') {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Run-Stage {
    param(
        [string]$Name,
        [string[]]$CargoArgs
    )

    Write-Log "===== Stage started: $Name ====="
    Write-Log "Command: cargo $($CargoArgs -join ' ')"

    $CommandLine = "cargo " + (($CargoArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ")
    & cmd.exe /d /c "$CommandLine 2>&1" | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        Write-Log "Stage failed: $Name. Exit code: $ExitCode"
        exit $ExitCode
    }

    Write-Log "Stage complete: $Name. Continuing to next stage."
}

Write-Log "Full generation started."
Write-Log "Date range: $StartDate..$EndDate"

Run-Stage `
    -Name "Label generation" `
    -CargoArgs @(
        "run", "--release",
        "--manifest-path", "factor_engine\Cargo.toml",
        "--",
        "label-run",
        "--asset", "stock",
        "--frequency", "daily",
        "--start-date", $StartDate,
        "--end-date", $EndDate
    )

Run-Stage `
    -Name "Barra SIZE generation" `
    -CargoArgs @(
        "run", "--release",
        "--manifest-path", "factor_engine\Cargo.toml",
        "--",
        "barra-run",
        "--asset", "stock",
        "--frequency", "daily",
        "--model", "CNE6",
        "--families", "SIZE",
        "--start-date", $StartDate,
        "--end-date", $EndDate
    )

Run-Stage `
    -Name "FZZQ factor generation" `
    -CargoArgs @(
        "run", "--release",
        "--manifest-path", "factor_engine\Cargo.toml",
        "--",
        "run",
        "--asset", "stock",
        "--frequency", "daily",
        "--start-date", $StartDate,
        "--end-date", $EndDate,
        "--tags", "FZZQ"
    )

Write-Log "All stages complete."
