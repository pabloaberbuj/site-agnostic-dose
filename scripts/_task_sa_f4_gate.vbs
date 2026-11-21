Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Program Files\Git\usr\bin\bash.exe"" -lc ""cd 'C:/Pablo/Site Agnostic/repo/site-agnostic-dose' && .venv/Scripts/python.exe scripts/f4_gate.py >> logs/f4_gate.out 2>&1""", 0, False
