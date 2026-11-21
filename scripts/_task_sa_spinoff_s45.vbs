Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Program Files\Git\usr\bin\bash.exe"" -lc ""cd 'C:/Pablo/Site Agnostic/repo/site-agnostic-dose' && bash scripts/spinoff_tradeoff/run_s45.sh > logs/run_spinoff_s45.out 2>&1""", 0, False
