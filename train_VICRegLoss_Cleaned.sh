#!/bin/bash


for ii in 0.1 0.5 1.0 10.;
 do for jj in 0. 0.1 1.0 10.;
    do 
      
      python3 IN_FlatSamples_VICRegLoss_Cleaned.py --weightstd ${ii} --weightrepr ${ii} --weightcov ${jj} --weightCorr1 0. --weightCorr2 0. --nepochs 20
 done
done
