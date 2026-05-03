param(
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [Parameter(Mandatory = $true)]
    [string]$Tokenizer,
    [string]$Prompt = "Write a Rust function to calculate factorial of a number.",
    [int]$SampleLen = 128,
    [double]$Temperature = 0,
    [double]$RepeatPenalty = 1.0,
    [string]$OutputDir = (Join-Path $PSScriptRoot "artifacts"),
    [switch]$Cpu,
    [switch]$Rebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-WithEnvironment {
    param(
        [hashtable]$Variables,
        [scriptblock]$Action
    )

    $previous = @{}
    foreach ($key in $Variables.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, [string]$Variables[$key], "Process")
    }

    try {
        & $Action
    }
    finally {
        foreach ($key in $Variables.Keys) {
            [Environment]::SetEnvironmentVariable($key, $previous[$key], "Process")
        }
    }
}

function Count-ApiCalls {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path $Path)) {
        return 0
    }

    return (Select-String -Path $Path -Pattern ("\b" + [regex]::Escape($Name) + "\b") -AllMatches | Measure-Object).Count
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$vkConfig = Get-Command vkconfig.exe -ErrorAction Stop
$sdkBin = Split-Path $vkConfig.Source -Parent
$gfxreconInfo = Join-Path $sdkBin "gfxrecon-info.exe"

if (-not (Test-Path $gfxreconInfo)) {
    throw "gfxrecon-info.exe not found in Vulkan SDK bin: $sdkBin"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

Push-Location $repoRoot
try {
    $exePath = Join-Path $repoRoot "target\release\examples\quantized-qwen3.exe"
    if ($Rebuild -or -not (Test-Path $exePath)) {
        cargo build --release --features vulkan --package candle-examples --example quantized-qwen3
    }
    if (-not (Test-Path $exePath)) {
        throw "quantized-qwen3.exe not found at $exePath"
    }

    $apiDumpFile = Join-Path $OutputDir "qwen3_apidump.txt"
    $apiDumpRunFile = Join-Path $OutputDir "qwen3_apidump_run.txt"
    $gfxrFile = Join-Path $OutputDir "qwen3_capture.gfxr"
    $gfxrRunFile = Join-Path $OutputDir "qwen3_gfxrecon_run.txt"
    $gfxrInfoFile = Join-Path $OutputDir "qwen3_gfxrecon_info.txt"
    $summaryFile = Join-Path $OutputDir "qwen3_vulkan_sdk_summary.txt"

    $appArgs = @(
        "--model", $Model,
        "--tokenizer", $Tokenizer,
        "--prompt", $Prompt,
        "--temperature", "$Temperature",
        "--sample-len", "$SampleLen",
        "--repeat-penalty", "$RepeatPenalty"
    )
    if ($Cpu) {
        $appArgs += "--cpu"
    }

    Invoke-WithEnvironment @{
        VK_INSTANCE_LAYERS      = "VK_LAYER_LUNARG_api_dump"
        VK_LAYER_PATH           = $sdkBin
        VK_APIDUMP_OUTPUT_FORMAT = "text"
        VK_APIDUMP_FLUSH        = "true"
    } {
        & $exePath @appArgs *>&1 | Tee-Object -FilePath $apiDumpFile | Tee-Object -FilePath $apiDumpRunFile | Out-Null
    }

    Invoke-WithEnvironment @{
        VK_INSTANCE_LAYERS              = "VK_LAYER_LUNARG_gfxreconstruct"
        VK_LAYER_PATH                   = $sdkBin
        GFXRECON_CAPTURE_FILE           = $gfxrFile
        GFXRECON_CAPTURE_FILE_TIMESTAMP = "false"
        GFXRECON_LOG_LEVEL              = "info"
    } {
        & $exePath @appArgs *>&1 | Tee-Object -FilePath $gfxrRunFile | Out-Null
    }

    if (-not (Test-Path $gfxrFile)) {
        throw "gfxreconstruct capture file was not created: $gfxrFile"
    }

    & $gfxreconInfo $gfxrFile | Tee-Object -FilePath $gfxrInfoFile | Out-Null

    $apiNames = @(
        "vkQueueSubmit",
        "vkBeginCommandBuffer",
        "vkEndCommandBuffer",
        "vkAllocateDescriptorSets",
        "vkCmdDispatch",
        "vkCmdCopyBuffer",
        "vkWaitForFences",
        "vkGetFenceStatus",
        "vkFlushMappedMemoryRanges"
    )

    $summaryLines = [System.Collections.Generic.List[string]]::new()
    $summaryLines.Add("Vulkan SDK profiling summary")
    $summaryLines.Add("Model: $Model")
    $summaryLines.Add("Tokenizer: $Tokenizer")
    $summaryLines.Add("Prompt: $Prompt")
    $summaryLines.Add("SampleLen: $SampleLen")
    $summaryLines.Add("Temperature: $Temperature")
    $summaryLines.Add("RepeatPenalty: $RepeatPenalty")
    $summaryLines.Add("")
    $summaryLines.Add("API dump counts:")
    foreach ($apiName in $apiNames) {
        $summaryLines.Add(("{0}`t{1}" -f $apiName, (Count-ApiCalls -Path $apiDumpFile -Name $apiName)))
    }
    $summaryLines.Add("")
    $summaryLines.Add("Artifacts:")
    $summaryLines.Add("  api dump: $apiDumpFile")
    $summaryLines.Add("  gfxrecon capture: $gfxrFile")
    $summaryLines.Add("  gfxrecon info: $gfxrInfoFile")

    Set-Content -Path $summaryFile -Value $summaryLines
    Get-Content $summaryFile
}
finally {
    Pop-Location
}
