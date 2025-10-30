@echo off
chcp 65001 >nul
echo ================================
echo   谣言检测系统自动启动脚本
echo ================================

echo [1/3] 激活Conda环境...
call conda activate sjmt
if %errorlevel% neq 0 (
    echo 错误: 无法激活sjmt环境，请检查环境是否存在
    pause
    exit /b 1
)

echo [2/3] 切换到项目目录...
cd /d "D:\UESTC\大四上\社交媒体分析\rumors_detection"
if %errorlevel% neq 0 (
    echo 错误: 无法切换到指定目录
    pause
    exit /b 1
)

echo [3/3] 启动Node.js服务器...
echo 服务器启动中，请勿关闭此窗口...
node server.js

REM 如果Node.js服务器退出，暂停以便查看错误
echo.
echo 服务器已停止，按任意键退出...
pause >nul