# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:28:23 2022

@author: jswee
"""
from joblib import Parallel, delayed

def myfun(arg):
     return arg+1

results = Parallel(n_jobs=-1, verbose=0, backend="threading")(
             map(delayed(myfun), range(5)))

print(results)