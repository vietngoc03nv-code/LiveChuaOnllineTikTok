@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title TikTok Live RPC
echo Dang khoi dong TikTok Live RPC...
node server.js
pause
