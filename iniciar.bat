@echo off
cd /d "%~dp0"
title AgroClima
cls

echo ============================================
echo  AgroClima - Sistema Agricola Inteligente
echo ============================================
echo.

if not exist "ngrok.exe" (
    echo [ERRO] ngrok.exe nao encontrado nesta pasta.
    echo Baixe em: https://ngrok.com/download
    echo Coloque ngrok.exe aqui: %CD%
    echo.
    pause
    exit /b 1
)

if not exist "data\climate_normals.pkl" (
    echo [1/4] Processando dados INMET...
    python api\preprocess.py
    echo.
    echo [2/4] Treinando modelo ML...
    python api\train_model.py
    echo.
) else (
    echo Dados prontos. Pulando preprocessamento.
    echo.
)

taskkill /IM ngrok.exe /F > nul 2>&1
timeout /t 1 /nobreak > nul

echo [3/4] Iniciando servidor...
start "AgroClima-API" /MIN cmd /c "cd /d %~dp0 && python api\main.py"
timeout /t 5 /nobreak > nul

echo [4/4] Iniciando ngrok...
start "AgroClima-ngrok" /MIN ngrok http 8000
timeout /t 6 /nobreak > nul

echo.
echo Obtendo URL publica...
powershell -NoProfile -Command "$url = (Invoke-RestMethod http://localhost:4040/api/tunnels).tunnels[0].public_url; Write-Host ''; Write-Host '============================================'; Write-Host '  SISTEMA ONLINE'; Write-Host '============================================'; Write-Host ''; Write-Host '  Local:   http://localhost:8000'; Write-Host ''; Write-Host '  Publico: ' $url; Write-Host ''; Set-Clipboard $url; Write-Host '  [OK] URL copiada! Cole com Ctrl+V'; Write-Host ''; Write-Host '============================================'"

start "" "http://localhost:8000"

echo.
echo Pressione qualquer tecla para ENCERRAR tudo.
pause > nul

taskkill /FI "WINDOWTITLE eq AgroClima-API" /T /F > nul 2>&1
taskkill /IM ngrok.exe /F > nul 2>&1
taskkill /IM python.exe /F > nul 2>&1
echo Encerrado.
