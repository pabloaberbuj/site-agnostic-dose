Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Program Files\Git\usr\bin\bash.exe"" -lc ""cd 'C:/Pablo/Site Agnostic/repo/site-agnostic-dose' && bash scripts/spinoff_tradeoff/run_1b_extract.sh > logs/run_spinoff_1b_x.out 2>&1""", 0, False
