function [ acc1,acc2,acc3 ] = runtimes( T )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
acc1 = 0;
acc2 = 0;
acc3 = 0;
for i = 1 : T
    fprintf(sprintf('test time :%d ', i));
    [a1,a2,a3]=cma();
    acc1 = a1/T + acc1;
    acc2 = a2/T + acc2;
    acc3 = a3/T + acc3;
end

end

