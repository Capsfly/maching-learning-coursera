A=[-2 6;7 4;5 0];
mean(A)
A=A-mean(A)
C=1/2*A'*A;
cov(A)
t1=[-5.33 3.66 1.66];
t2=[2.67 0.67 -3.33]';
t3=t2*t1;
a=mean(t3,'all');
