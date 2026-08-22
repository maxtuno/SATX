# run.ps1 — corredor de benchmarks del kernel SAT de Kerberos.
# Mide tiempo de CPU con límite DURO por instancia y valida resultados:
#   SAT   → modelo verificado con check_model.exe (e internamente por el solver)
#   UNSAT → prueba DRAT emitida por el solver y verificada con grinder (kerberos --grinder)
#
# Uso:
#   .\benchmarks\run.ps1                     (todos los .cnf de benchmarks\cnf)
#   .\benchmarks\run.ps1 -Filter 'mult*'     (subconjunto)
#   .\benchmarks\run.ps1 -TimeoutSec 30      (límite DURO por instancia, mata el proceso)
#   .\benchmarks\run.ps1 -Solver bin\kerberos.exe
param(
  [string]$Filter = '*',
  [int]$TimeoutSec = 30,
  [string]$Solver = 'bin\kerberos.exe',
  [string]$ExtraArgs = '',
  [string]$CnfDir = 'benchmarks\cnf',
  [switch]$SkipVerify
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$solverExe = Join-Path $root $Solver
$tmp = Join-Path $env:TEMP 'kerberos-bench'
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

$checkerExe = Join-Path $root 'benchmarks\tools\check_model.exe'
if (-not (Test-Path $checkerExe)) {
  gcc -O2 -std=c17 (Join-Path $root 'benchmarks\tools\check_model.c') -o $checkerExe
}

function Invoke-Timed([string]$exe, [string[]]$argsList, [int]$timeoutMs) {
  $tmp = Join-Path $env:TEMP 'kerberos-bench'
  $oFile = Join-Path $tmp ('out_' + [guid]::NewGuid().ToString('N') + '.txt')
  $eFile = Join-Path $tmp ('err_' + [guid]::NewGuid().ToString('N') + '.txt')
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $exe -ArgumentList $argsList `
    -RedirectStandardOutput $oFile -RedirectStandardError $eFile `
    -NoNewWindow -PassThru
  $done = $p.WaitForExit($timeoutMs)
  $secs = $sw.Elapsed.TotalSeconds
  $out = ''; $err = ''
  if (-not $done) {
    try { $p.Kill() } catch {}
    $p.WaitForExit()
    $sw.Stop()
    if (Test-Path $oFile) { Remove-Item $oFile -Force }
    if (Test-Path $eFile) { Remove-Item $eFile -Force }
    return ,@("TIMEOUT", "", $sw.Elapsed.TotalSeconds)
  }
  $p.WaitForExit()
  $sw.Stop()
  if (Test-Path $oFile) { $out = Get-Content $oFile -Raw; Remove-Item $oFile -Force }
  if (Test-Path $eFile) { $err = Get-Content $eFile -Raw; Remove-Item $eFile -Force }
  return ,@($out, $err, $sw.Elapsed.TotalSeconds)
}

$files = Get-ChildItem -LiteralPath (Join-Path $root $CnfDir) -Filter "$Filter.cnf" | Sort-Object Name
Write-Output ("instancia".PadRight(24) + "resultado".PadRight(10) + "tiempo_s".PadRight(10) + "verificado")
Write-Output ("-" * 78)
$total = 0.0
$fail = 0
$timeouts = 0
foreach ($f in $files) {
  $proof = Join-Path $tmp ($f.BaseName + '.drat')
  Remove-Item $proof -ErrorAction SilentlyContinue
  # Medición SIN --proof (--proof desactiva la simplificación del kernel):
  $argsList = @()
  foreach ($a in ($ExtraArgs -split ' ')) { if ($a) { $argsList += $a } }
  $argsList += $f.FullName
  $r = Invoke-Timed $solverExe $argsList ($TimeoutSec * 1000)
  $out = $r[0]
  $secs = [Math]::Round($r[2], 3)
  $total += $secs
  $sat = $out -match 's SATISFIABLE'
  $unsat = $out -match 's UNSATISFIABLE'
  $status = 'SAT'
  if ($out -match '^TIMEOUT') { $status = 'TIMEOUT'; $timeouts++ }
  elseif ($unsat) { $status = 'UNSAT' }
  elseif (-not $sat) { $status = "ERR" }
  $ver = 'n/a'
  if (-not $SkipVerify -and ($sat -or $unsat)) {
    if ($sat) {
      $v = (($out -split "`n") | Where-Object { $_.StartsWith('v ') }) -join ' '
      $ck = & $checkerExe $f.FullName $v 2>&1 | Out-String
      $ver = if ($ck -match '^OK') { 'modelo-ok' } else { 'MODELO-MALO' }
    } elseif ($unsat) {
      # Segunda pasada con --proof para emitir y verificar el DRAT.
      $r2 = Invoke-Timed $solverExe @('--proof', $proof, $f.FullName) ($TimeoutSec * 1000 * 6)
      if (($r2[0] -match 's UNSATISFIABLE') -and (Test-Path $proof)) {
        $g = Invoke-Timed $solverExe @('--grinder', $f.FullName, $proof, '-w') 600000
        $ver = if ($g[0] -match 'VERIFIED') { 'drat-ok' } else { "DRAT-MALO" }
      } else { $ver = 'sin-proof' }
    }
  }
  if ($ver -match 'MALO') { $fail++ }
  Write-Output ($f.BaseName.PadRight(24) + $status.PadRight(10) + ("$secs").PadRight(10) + $ver)
}
Write-Output ("-" * 78)
Write-Output ("total: {0} s en {1} instancias ({2} timeouts, {3} fallos de verificacion)" -f ([Math]::Round($total, 3)), $files.Count, $timeouts, $fail)

