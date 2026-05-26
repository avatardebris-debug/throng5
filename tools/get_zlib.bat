@echo off
REM ============================================================
REM  Install zlib for Windows (needed by stable-retro CMake)
REM  Downloads prebuilt zlib headers + .lib from zlib official
REM ============================================================
cd /d C:\Users\avata\aicompete

if not exist zlib-win64 (
    echo Downloading zlib prebuilt for Windows x64...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/nicowillis/zlib-windows-x64/releases/download/1.3.1/zlib-1.3.1-win64.zip' -OutFile zlib-win64.zip" 2>&1
    if not exist zlib-win64.zip (
        echo Primary mirror failed, trying alternate...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.zlib.net/zlib131.zip' -OutFile zlib-win64.zip" 2>&1
    )
    powershell -Command "Expand-Archive -Path zlib-win64.zip -DestinationPath zlib-win64 -Force"
)

echo Zlib directory:
dir C:\Users\avata\aicompete\zlib-win64
