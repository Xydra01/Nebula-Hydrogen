# Registers a scheduled task to start nebula-daemon automatically at logon (Issue #38).
# Run from an unprivileged or elevated PowerShell as the interactive user.
# Re-running updates/replaces the task. Use -Unregister to remove it.
param(
    [switch]$Unregister,
    [int]$DelaySeconds = 30
)
$ErrorActionPreference = 'Stop'

$taskPath = '\Nebula\'
$taskName = 'Nebula daemon (logon)'

if ($Unregister) {
    $existing = Get-ScheduledTask -TaskPath $taskPath -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskPath $taskPath -TaskName $taskName -Confirm:$false
        Write-Host "Unregistered $taskPath$taskName successfully."
    } else {
        Write-Host "Task $taskPath$taskName does not exist. Nothing to unregister."
    }
    return
}

$nebula = 'F:\Nebula\bin\nebula.exe'
if (-not (Test-Path $nebula)) {
    throw "$nebula not found; build and run scripts\install-nebula.ps1 first"
}

$user = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

# StartWhenAvailable: catch up if missed
# DontStopIfGoingOnBatteries / AllowStartIfOnBatteries: keep laptop users covered
# MultipleInstances IgnoreNew: do not spawn duplicates
# ExecutionTimeLimit 0: daemon runs continuously without timeout kill
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -Hidden

# Logon trigger with delay to avoid competing with desktop startup
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
if ($DelaySeconds -gt 0) {
    $trigger.Delay = "PT${DelaySeconds}S"
}

# Headless conhost keeps console window from popping up
$action = New-ScheduledTaskAction `
    -Execute 'conhost.exe' `
    -Argument "--headless `"$nebula`" daemon start" `
    -WorkingDirectory 'F:\Nebula\bin'

Register-ScheduledTask `
    -TaskName $taskName `
    -TaskPath $taskPath `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force `
    -ErrorAction Stop | Out-Null

Write-Host "Registered $taskPath$taskName"
Write-Host "Trigger: User logon for $user with ${DelaySeconds}s delay"
Write-Host "Action: conhost.exe --headless `"$nebula`" daemon start"
Write-Host "Run 'nebula daemon status' after logon to verify readiness."
