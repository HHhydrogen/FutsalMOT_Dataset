#Requires -Version 7.0
<#
默认仅打印计划。主调度确认 UE 渲染全部结束后，可用 All 一次顺序执行完整流程。
Gpu 阶段拒绝任何仍存在的 UnrealEditor 进程；本脚本不关闭 Editor，不启动 UE。
每条命令同步执行，失败立即停止。真实 test 和可选 D 均不在本轮计划中。
#>
[CmdletBinding()]
param(
    [ValidateSet('Plan', 'Readiness', 'Prepare', 'Gpu', 'All')]
    [string]$Phase = 'Plan',
    [string]$Python = 'D:/projects/FutsalMOT/.venv/Scripts/python.exe',
    [string]$CanonicalRoot = 'G:/FutsalMOT_Dataset/v1_no_numbers_4k',
    [string]$ResearchRoot = 'D:/projects/FutsalMOT_Research'
)
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot 'pipeline.py'
$protocol = Join-Path $PSScriptRoot 'paper_pilot_protocol.json'
$aName = 'pretrained_val_gpu_seed20261021'
$bName = 'real_only_val_gpu_seed20261021_e5'
$cName = 'real_plus_synthetic_val_gpu_seed20261021_e5'
$aWeights = Join-Path $ResearchRoot 'weights/yolo11n.pt'
$bWeights = Join-Path $ResearchRoot 'runs/real_only_seed20261021_e5/weights/last.pt'
$cWeights = Join-Path $ResearchRoot 'runs/real_plus_synthetic_seed20261021_e5/weights/last.pt'

function Assert-NoUnreal {
    $running = @(Get-Process -Name 'UnrealEditor*' -ErrorAction SilentlyContinue)
    if ($running.Count -gt 0) {
        throw 'UE 进程仍存在。等待全部渲染结束并由主调度安全关闭 Editor 后再运行 Gpu。'
    }
}

function Invoke-Step {
    param([string[]]$Arguments, [switch]$Gpu)
    Write-Host ($Python + ' -X utf8 ' + $entry + ' ' + ($Arguments -join ' '))
    if ($Phase -eq 'Plan') { return }
    if ($Gpu) { Assert-NoUnreal }
    & $Python -X utf8 $entry @Arguments
    if ($LASTEXITCODE -ne 0) { throw "实验步骤失败，exit=$LASTEXITCODE。停止后续步骤。" }
}

function Invoke-Val {
    param([string]$Name, [string]$Weights)
    Invoke-Step -Gpu -Arguments @('evaluate','--device','0','--split','val','--imgsz','640','--batch','4','--name',$Name,'--weights',$Weights)
    Invoke-Step -Gpu -Arguments @('track','--device','0','--split','val','--imgsz','640','--name',$Name,'--weights',$Weights)
    Invoke-Step -Arguments @('evaluate-tracks','--split','val','--name',$Name)
}

if ($Phase -ne 'Plan' -and $ResearchRoot.Replace('\','/').TrimEnd('/') -ne 'D:/projects/FutsalMOT_Research') {
    throw 'pipeline 的研究根目录固定；此计划仅接受已核验的 D:/projects/FutsalMOT_Research。'
}
if ($Phase -in @('Gpu','All')) { Assert-NoUnreal }

# 所有实际阶段先只读验证八个完整发布组、完整 checksum、标定与选择协议。
Invoke-Step -Arguments @('verify-protocol','--canonical-root',$CanonicalRoot,'--protocol',$protocol,'--verification-level','structure')
if ($Phase -eq 'Readiness') { return }

if ($Phase -in @('Plan','Prepare','All')) {
    # 不重建真实数据、不重建 B；跨卷 RGB 缓存由 YOLO/MOT 共享。
    Invoke-Step -Arguments @('prepare-synthetic','--canonical-root',$CanonicalRoot,'--verification-level','structure','--reference-images')
    Invoke-Step -Arguments @('prepare-synthetic-mot','--canonical-root',$CanonicalRoot,'--verification-level','structure','--reference-images')
    Invoke-Step -Arguments @('prepare-arms','--seed','20261021','--protocol',$protocol)
    if ($Phase -eq 'Prepare') { return }
}

if ($Phase -in @('Gpu','All')) {
    foreach ($name in @($aName,$bName,$cName)) {
        foreach ($folder in @('evaluations','tracking','tracking_metrics')) {
            if (Test-Path -LiteralPath (Join-Path $ResearchRoot "$folder/$name")) {
                throw "输出已存在：$folder/$name。保留已有结果，从未完成步骤继续，不覆盖或静默续训。"
            }
        }
    }
    if (Test-Path -LiteralPath (Join-Path $ResearchRoot 'runs/real_plus_synthetic_seed20261021_e5')) {
        throw 'C 训练输出已存在；先核验 receipt，再从对应未完成评测步骤继续。'
    }
    if (!(Test-Path -LiteralPath (Join-Path $ResearchRoot 'datasets/arm_real_plus_synthetic/freeze.json'))) {
        throw 'C 尚未冻结；先完成 Prepare 阶段。'
    }
}

# A/B 都用新名称在同一 GPU 重评 real_val，保留历史 CPU/GPU 结果原样。
Invoke-Step -Arguments @('verify-arms','--protocol',$protocol)
Invoke-Val -Name $aName -Weights $aWeights
Invoke-Val -Name $bName -Weights $bWeights
Invoke-Step -Gpu -Arguments @('train','--device','0','--arm','real_plus_synthetic','--epochs','5','--batch','4','--imgsz','640','--seed','20261021')
Invoke-Val -Name $cName -Weights $cWeights
Invoke-Step -Arguments @('compare','--split','val','--names',$aName,$bName,$cName)
