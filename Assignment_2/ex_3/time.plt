set xlabel 'number of particles'
set ylabel 'execution time per particle (s)'
set logscale x


#plot 'time.dat' using 1:7 with linespoints title 'CPU',\
#'time.dat' using 1:2 with linespoints title 'GPU blocksize 16',\
#'time.dat' using 1:3 with linespoints title 'GPU blocksize 32',\
#'time.dat' using 1:4 with linespoints title 'GPU blocksize 64',\
#'time.dat' using 1:5 with linespoints title 'GPU blocksize 128',\
#'time.dat' using 1:6 with linespoints title 'GPU blocksize 256'

plot \
'time.dat' using 1:($2/$1) with linespoints title 'GPU blocksize 16',\
'time.dat' using 1:($3/$1) with linespoints title 'GPU blocksize 32',\
'time.dat' using 1:($4/$1) with linespoints title 'GPU blocksize 64',\
'time.dat' using 1:($5/$1) with linespoints title 'GPU blocksize 128',\
'time.dat' using 1:($6/$1) with linespoints title 'GPU blocksize 256'

