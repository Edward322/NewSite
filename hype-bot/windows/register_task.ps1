# Registers the bot in Windows Task Scheduler:
#   - start at boot (1 minute delay) and at logon; runs whether the user is logged on or not (S4U)
#   - restart on failure every minute, no time limit, works on battery
#   - one instance only
param([string]$Mode = "demo", [string]$BotDir)
$ErrorActionPreference = "Stop"
$name = "HypeBot-$Mode"
$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" `
  -Argument "/c `"`"$BotDir\windows\run_bot.bat`" $Mode`"" -WorkingDirectory $BotDir
$boot = New-ScheduledTaskTrigger -AtStartup
$boot.Delay = "PT1M"
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
try {
  $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
  Register-ScheduledTask -TaskName $name -Action $action -Trigger $boot, $logon -Settings $settings `
    -Principal $principal -Force | Out-Null
  Write-Host "Task $name registered: starts at boot, even without logon."
} catch {
  Write-Host "S4U not allowed here ($($_.Exception.Message)); registering for logged-on user."
  $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $name -Action $action -Trigger $logon -Settings $settings `
    -Principal $principal -Force | Out-Null
  Write-Host "Task $name registered: starts when you log on to Windows."
  Write-Host "To start after a power failure without you, enable automatic logon (netplwiz)."
}
