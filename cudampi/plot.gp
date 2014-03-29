set view 71, 319

#hides connection lines drawn in order of appearance in file
set hidden3d
set surface
set ticslevel 0.01
set yrange [767:1024]
set xrange [767:1024]
set xlabel "Blocks"
set ylabel "Threads"
splot "times.dat" using 1:2:($3/($1*$2)) title "Execution Time per Process in ms" with lines

pause -1

set terminal png size 2560,1980
set output "timesHighEndPerProcess.png"
set view 71, 319
splot "times.dat" using 1:2:($3/($1*$2)) title "Execution Time per Process in ms" with lines

