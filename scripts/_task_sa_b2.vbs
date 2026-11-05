Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Program Files\Git\usr\bin\bash.exe"" -lc ""cd 'C:/Pablo/Site Agnostic/repo/site-agnostic-dose' && TRAIN_W=1 bash scripts/run_b2.sh >> logs/run_b2.out 2>&1""", 0, False
