set term wxt font "DejaVu Serif"
set title "Scaling Behaviour of CUDA MD5 Hashrate"
set xlabel "Blocks"

set ylabel "Time/Block/ms"
plot "Benchmark/times1Block.dat"  using 1:($2/ 1):3 title "x Threads,  1 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times2Block.dat"  using 1:($2/ 2):3 title "x Threads,  2 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times4Block.dat"  using 1:($2/ 4):3 title "x Threads,  4 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times6Block.dat"  using 1:($2/ 6):3 title "x Threads,  6 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times7Block.dat"  using 1:($2/ 7):3 title "x Threads,  7 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times8Block.dat"  using 1:($2/ 8):3 title "x Threads,  8 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times9Block.dat"  using 1:($2/ 9):3 title "x Threads,  9 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times13Block.dat" using 1:($2/13):3 title "x Threads, 13 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times16Block.dat" using 1:($2/16):3 title "x Threads, 16 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times31Block.dat" using 1:($2/31):3 title "x Threads, 31 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times32Block.dat" using 1:($2/32):3 title "x Threads, 32 Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times33Block.dat" using 1:($2/33):3 title "x Threads, 33 Blocks" pt 1 pointsize 1 lw 1
pause -1
set ylabel "Time/Thread/ms"
plot "Benchmark/times1Threads.dat"  using 1:($2/ 1):3 title " 1 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times5Threads.dat"  using 1:($2/ 5):3 title " 5 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times8Threads.dat"  using 1:($2/ 8):3 title " 8 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times12Threads.dat" using 1:($2/12):3 title "12 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times13Threads.dat" using 1:($2/13):3 title "13 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times31Threads.dat" using 1:($2/31):3 title "31 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times32Threads.dat" using 1:($2/32):3 title "32 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times33Threads.dat" using 1:($2/33):3 title "33 Threads, x Blocks" pt 1 pointsize 1 lw 1,\
     "Benchmark/times35Threads.dat" using 1:($2/35):3 title "35 Threads, x Blocks" pt 1 pointsize 1 lw 1
pause -1

################### SAVE TO SVG ###################

set term svg font "DejaVu Serif"
set term svg lw 0.1
set output "timesBlocks.svg"

plot "Benchmark/times1Block.dat"  using 1:2:3 title "x Threads,  1 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times2Block.dat"  using 1:2:3 title "x Threads,  2 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times4Block.dat"  using 1:2:3 title "x Threads,  4 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times6Block.dat"  using 1:2:3 title "x Threads,  6 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times7Block.dat"  using 1:2:3 title "x Threads,  7 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times8Block.dat"  using 1:2:3 title "x Threads,  8 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times9Block.dat"  using 1:2:3 title "x Threads,  9 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times13Block.dat" using 1:2:3 title "x Threads, 13 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times16Block.dat" using 1:2:3 title "x Threads, 16 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times31Block.dat" using 1:2:3 title "x Threads, 31 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times32Block.dat" using 1:2:3 title "x Threads, 32 Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times33Block.dat" using 1:2:3 title "x Threads, 33 Blocks" pt 1 pointsize 0.5 lw 0.5

set output "timesThreads.svg"

plot "Benchmark/times1Threads.dat"  using 1:2:3 title " 1 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times5Threads.dat"  using 1:2:3 title " 5 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times8Threads.dat"  using 1:2:3 title " 8 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times12Threads.dat" using 1:2:3 title "12 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times13Threads.dat" using 1:2:3 title "13 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times31Threads.dat" using 1:2:3 title "31 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times32Threads.dat" using 1:2:3 title "32 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times33Threads.dat" using 1:2:3 title "33 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5,\
     "Benchmark/times35Threads.dat" using 1:2:3 title "35 Threads, x Blocks" pt 1 pointsize 0.5 lw 0.5

exit
