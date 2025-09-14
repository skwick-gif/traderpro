@echo off
   echo Select option:
   echo 1. Push to Git
   echo 2. Pull from Git
   set /p choice=Enter choice (1 or 2):

   if "%choice%"=="1" (
     echo Checking status...
     git status
     echo Adding all files...
     git add .
     echo Committing with message: "General updates"...
     git commit -m "General updates"
     echo Pushing to main...
     git push origin main
   ) else if "%choice%"=="2" (
     echo Pulling from main...
     git pull origin main --rebase
   ) else (
     echo Invalid choice!
   )
   echo Done! Press any key to exit...
   pause