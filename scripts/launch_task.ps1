# Lanza un comando bash desacoplado de VS Code / Claude Code como Scheduled Task de Windows (sobrevive al cierre del editor).
# NO sobrevive a logoff / suspensión / hibernación del equipo. Reanudable: los trains usan --resume (last.ckpt).
#   Programar de noche: ... -At 22:00
#   powershell -File scripts/launch_task.ps1 -Name sa_b_chain -Cmd "bash scripts/run_b_chain.sh > logs/run_b_chain.out 2>&1"
#   Estado: schtasks /query /tn sa_b_chain /v /fo list   |   Parar: schtasks /end /tn sa_b_chain   |   Borrar: schtasks /delete /tn sa_b_chain /f
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Cmd,
    [string]$At = ''     # 'HH:mm' -> arranca a esa hora (hoy, o mañana si ya pasó); vacío = arranca ya
)
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bash = 'C:\Program Files\Git\usr\bin\bash.exe'
# Sin ventana de consola: bash.exe directo abre un cmd que cualquiera puede cerrar (exit 0xC000013A) y mata el entrenamiento.
# wscript con WindowStyle 0 lo corre oculto; no hay ventana que cerrar.
$bashCmd = "`"$bash`" -lc `"cd '$($repo -replace '\\','/')' && $Cmd`""
$vbs = Join-Path $PSScriptRoot "_task_$Name.vbs"
$q = [string][char]34
$vbsCmd = $bashCmd.Replace($q, $q + $q)
Set-Content -Path $vbs -Encoding ASCII -Value @('Set sh = CreateObject("WScript.Shell")', ('sh.Run ' + $q + $vbsCmd + $q + ', 0, False'))
$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "//B //Nologo `"$vbs`"" -WorkingDirectory $repo
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $Name -Confirm:$false }
Register-ScheduledTask -TaskName $Name -Action $action -Settings $settings -Principal $principal | Out-Null
if ($At) {
    $when = [datetime]::Today.Add([TimeSpan]::Parse($At))
    if ($when -le (Get-Date)) { $when = $when.AddDays(1) }
    Set-ScheduledTask -TaskName $Name -Trigger (New-ScheduledTaskTrigger -Once -At $when) | Out-Null
    "Programada para $when"
} else { Start-ScheduledTask -TaskName $Name }
Get-ScheduledTask -TaskName $Name | Select-Object TaskName, State | Format-Table -AutoSize
