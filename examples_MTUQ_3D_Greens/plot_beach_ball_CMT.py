#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:36:49 2020

@author: andrei
"""

from obspy.imaging.beachball import beachball


#mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
M6 = [  5.16690308e+14,  -2.93574118e+15,   2.41905087e+15,        -1.34301906e+14,  -2.96918618e+15,  -5.82373966e+15]
#M6 = [-2.64209032e+24,  5.28142928e+23,  2.36085327e+24,  5.08171674e+22, -1.86496587e+24, -3.72439542e+23]
Mrr =  M6[0];Mtt =  M6[2]; Mpp =  M6[1]; Mrt =  M6[4]; Mrp = M6[3]; Mpt = M6[5];
#mt = [Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]
mt = [Mrr,Mpp,Mtt,Mrp,Mrt,Mpt]
beachball(mt, size=200, linewidth=2, facecolor='b')

#mt2 = [150, 87, 1]
#mt2 = [90, 45, 90]
#mt2 = [252, 82, 150]
mt2 = [263, 	84, 	-163]
beachball(mt2, size=200, linewidth=2, facecolor='r')

#mt3 = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
mt3 = [  5.16690308e+14,  -2.93574118e+15,   2.41905087e+15,        -1.34301906e+14,  -2.96918618e+15,  -5.82373966e+15]
#mt3 = [-2.64209032e+24,  5.28142928e+23,  2.36085327e+24,  5.08171674e+22, -1.86496587e+24, -3.72439542e+23]
beachball(mt3, size=200, linewidth=2, facecolor='g')