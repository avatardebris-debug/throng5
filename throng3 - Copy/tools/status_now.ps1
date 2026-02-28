param()

$ErrorActionPreference = 'SilentlyContinue'
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== THRONG STATUS NOW ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Host "Repo: $repoRoot"
Write-Host ""

# 1) NOW.md quick view
$nowPath = Join-Path $repoRoot 'NOW.md'
if (Test-Path $nowPath) {
    Write-Host "--- NOW.md ---" -ForegroundColor Yellow
    Get-Content $nowPath | Select-Object -First 40
    Write-Host ""
} else {
    Write-Host "NOW.md not found at $nowPath" -ForegroundColor Red
}

# 2) Git summary
Write-Host "--- Git ---" -ForegroundColor Yellow
$branch = git -C $repoRoot rev-parse --abbrev-ref HEAD
$head = git -C $repoRoot log -1 --pretty=format:"%h %ad %s" --date=short
Write-Host "Branch: $branch"
Write-Host "HEAD:   $head"
$dirty = git -C $repoRoot status --short
if ($dirty) {
    Write-Host "Dirty files:" -ForegroundColor Red
    $dirty
} else {
    Write-Host "Working tree clean." -ForegroundColor Green
}
Write-Host ""

# 3) Latest blind_hypothesis_log row via Python sqlite3
Write-Host "--- blind_hypothesis_log (latest) ---" -ForegroundColor Yellow
$dbPath = Join-Path $repoRoot 'experiments\experiments.db'
if (Test-Path $dbPath) {
    $py = @"
import sqlite3
p = r'''$dbPath'''
con = sqlite3.connect(p)
cur = con.cursor()
try:
    row = cur.execute('SELECT ts, blind_label, total, valid_count, gen_universal, gen_class, gen_instance FROM blind_hypothesis_log ORDER BY ts DESC LIMIT 1').fetchone()
    if row:
        print('ts=%s | label=%s | total=%s | valid=%s | U=%s C=%s I=%s' % row)
    else:
        print('No rows in blind_hypothesis_log')
except Exception as e:
    print(f'Query unavailable: {e}')
finally:
    con.close()
"@
    python -c $py
} else {
    Write-Host "DB not found: $dbPath" -ForegroundColor DarkYellow
}
Write-Host ""

# 4) Latest request/response artifacts
Write-Host "--- Latest memory artifacts ---" -ForegroundColor Yellow
$memoryDir = Join-Path $HOME '.openclaw\workspace\memory'
if (Test-Path $memoryDir) {
    $latestReq = Get-ChildItem -Path $memoryDir -Filter 'hyp_request_*.md' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $latestOut = Get-ChildItem -Path $memoryDir -Filter 'hypotheses_*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($latestReq) { Write-Host "Request : $($latestReq.FullName)" }
    else { Write-Host "Request : none" }

    if ($latestOut) { Write-Host "Response: $($latestOut.FullName)" }
    else { Write-Host "Response: none" }
} else {
    Write-Host "Memory dir not found: $memoryDir" -ForegroundColor DarkYellow
}
