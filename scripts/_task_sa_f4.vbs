Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Program Files\Git\usr\bin\bash.exe"" -lc ""cd 'C:/Pablo/Site Agnostic/repo/site-agnostic-dose' && bash scripts/run_f4.sh >> logs/run_f4.out 2>&1""", 0, False
