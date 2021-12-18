# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 18:11:03 2021

@author: jswee
"""

#!/usr/bin/env python3
import sys
assert sys.version_info >= (3,0), "please, use Python version 3"

import numpy as np
import matplotlib.pyplot as plt

# import SIM5 library
import sim5

# make array of spin values from 0 to 1
bh_spin = np.linspace(0, 1, 200)

# compute 
# - black-hole event horizon radius,
# - marginally stable orbit radius,
# - marginally bound radius,
# - photon orbit radius
# for all spins in bh_spin array
rbh = list(map(sim5.r_bh, bh_spin))
rms = list(map(sim5.r_ms, bh_spin))
rmb = list(map(sim5.r_mb, bh_spin))
rph = list(map(sim5.r_ph, bh_spin))

# make a plot of rbh and rms as a function of spin
plt.title("Kerr orbits")
plt.plot(bh_spin, rms, label='marg. stable orbit')
plt.plot(bh_spin, rmb, label='marg. bound orbit')
plt.plot(bh_spin, rph, label='photon orbit')
plt.plot(bh_spin, rbh, label='event horizon')
plt.xlabel(r'BH spin')
plt.ylabel(r'radius [$r_{\rm g}$]')
plt.legend()

plt.show()