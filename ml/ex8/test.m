YEAR=2010:2017;
[x,y]=meshgrid(1,YEAR);
plot3(x,YEAR,Kentucky);
hold on;
plot3(x+1,YEAR,Ohio);
plot3(x+2,YEAR,PA);
plot3(x+3,YEAR,Virginia);



