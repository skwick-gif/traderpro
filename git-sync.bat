@echo off
setlocal enabledelayedexpansion

rem 1) ודא שאנחנו בתוך ריפו
git rev-parse --is-inside-work-tree >nul 2>&1 || (
  echo Not a git repository. Run this from your project folder.
  exit /b 1
)

rem 2) צור .gitignore בסיסי אם חסר
if not exist ".gitignore" (
  echo node_modules/>> .gitignore
  echo dist/>> .gitignore
  echo build/>> .gitignore
  echo .env>> .gitignore
  echo .DS_Store>> .gitignore
)

rem 3) הורד node_modules מהאינדקס למקרה שהסתנן
if exist "node_modules" (
  git rm -r --cached node_modules >nul 2>&1
)

rem 4) הבא עדכונים והישאר על main (לא חובה לשנות אם יש לך ברנצ' אחר)
git fetch origin
git rev-parse --verify main >nul 2>&1 && git checkout main
git pull --rebase origin main

rem 5) הוסף הכל לפי .gitignore (בלי --force)
git add -A

rem 6) קומיט (עם הודעה ברירת מחדל אם לא ניתנה)
set msg=%*
if "%msg%"=="" set msg=sync: update
git commit -m "%msg%" || echo Nothing to commit.

rem 7) דחיפה
git push origin HEAD
