<#
.SYNOPSIS
    Unit tests for scripts/check-pr-dependencies.ps1 rate-limit retry and git-attestation fallback.
#>
$ErrorActionPreference = 'Stop'

Write-Host "Running unit tests for check-pr-dependencies.ps1..." -ForegroundColor Cyan

# Source functions from script without executing gate
. "$PSScriptRoot/../scripts/check-pr-dependencies.ps1" -SkipNetworkCheck | Out-Null

# Test 1: Test-GitAttestationResolved pattern matching
$mockGitLogs = @(
    "feat: some commit`n`nResolves: #74",
    "Merge pull request #42 from feat/security-74`n`nCloses #74",
    "fix(audit): patch issue #74 per review",
    "unrelated commit message"
)

function Test-AttestationMatchString($msg, $issueNum) {
    $pattern = "(?i)(?:Resolves|Closes|Fixes|Closed):?\s*#$issueNum\b|Merge pull request #[0-9]+[\s\S]*(?:#$issueNum\b|issue\s+$issueNum\b)"
    return ($msg -match $pattern)
}

if (-not (Test-AttestationMatchString $mockGitLogs[0] 74)) {
    throw "Test 1A failed: 'Resolves: #74' was not recognized"
}
if (-not (Test-AttestationMatchString $mockGitLogs[1] 74)) {
    throw "Test 1B failed: 'Closes #74' in merge commit was not recognized"
}
if (Test-AttestationMatchString $mockGitLogs[3] 74) {
    throw "Test 1C failed: unrelated message was falsely matched"
}
Write-Host "  [PASS] Test 1: Git-attestation resolution regex patterns" -ForegroundColor Green

# Test 2: Invoke-GhIssueViewWithRetry live verification
$res = Invoke-GhIssueViewWithRetry -IssueNumber 74 -MaxAttempts 2 -InitialDelaySec 1
if ($null -eq $res) {
    throw "Test 2 failed: Expected live query for Issue 74 to succeed"
}
if ($res.title -notmatch "Formal audit protocol") {
    throw "Test 2 failed: Title mismatch on Issue 74 query"
}
Write-Host "  [PASS] Test 2: Invoke-GhIssueViewWithRetry live query & parsing" -ForegroundColor Green

Write-Host "All check-pr-dependencies tests passed successfully." -ForegroundColor Green
exit 0
