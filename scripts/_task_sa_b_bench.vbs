Set sh = CreateObject("WScript.Shell")
sh.Run """C:\Program Files\Git\usr\bin\bash.exe"" -lc ""cd 'C:/Pablo/Site Agnostic/repo/site-agnostic-dose' && VARIANTS='3 4' bash scripts/run_b_bench.sh > logs/run_b_bench34.out 2>&1""", 0, False
