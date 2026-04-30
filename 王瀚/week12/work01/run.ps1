# NL2SQL Agent 快速启动脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Chinook 数据库 NL2SQL Agent 快速启动" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查环境变量
if ($env:DASHSCOPE_API_KEY) {
    $masked_key = $env:DASHSCOPE_API_KEY.Substring(0, 8) + "..." + $env:DASHSCOPE_API_KEY.Substring($env:DASHSCOPE_API_KEY.Length - 4)
    Write-Host "[✓] API Key 已设置: $masked_key" -ForegroundColor Green
} else {
    Write-Host "[✗] 未设置 DASHSCOPE_API_KEY 环境变量" -ForegroundColor Red
    Write-Host ""
    Write-Host "请先设置环境变量:" -ForegroundColor Yellow
    Write-Host "  `$env:DASHSCOPE_API_KEY='your-api-key-here'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "或者创建 .env 文件（参考 .env.example）" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "请选择运行模式:" -ForegroundColor Cyan
Write-Host "  1. 测试模式（运行预设问题）" -ForegroundColor White
Write-Host "  2. 演示模式（展示更多示例）" -ForegroundColor White
Write-Host "  3. 交互模式（自由提问）" -ForegroundColor White
Write-Host ""

$choice = Read-Host "请输入选项 (1/2/3)"

Write-Host ""
switch ($choice) {
    "1" {
        Write-Host "启动测试模式..." -ForegroundColor Green
        Write-Host ""
        python nl2sql_agent.py
    }
    "2" {
        Write-Host "启动演示模式..." -ForegroundColor Green
        Write-Host ""
        python nl2sql_interactive.py --demo
    }
    "3" {
        Write-Host "启动交互模式..." -ForegroundColor Green
        Write-Host ""
        python nl2sql_interactive.py
    }
    default {
        Write-Host "无效的选项" -ForegroundColor Red
    }
}
