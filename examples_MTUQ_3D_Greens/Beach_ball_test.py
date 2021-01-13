#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:36:49 2020

@author: andrei
"""

from obspy.imaging.beachball import beachball


#mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]

M6 = [-2.64209032e+24,  5.28142928e+23,  2.36085327e+24,  5.08171674e+22, -1.86496587e+24, -3.72439542e+23]
Mrr =  M6[2];Mtt =  M6[0]; Mpp =  M6[1]; Mrt =  M6[4]; Mrp = -M6[5]; Mtp = -M6[3];
mt = [Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]
beachball(mt, size=200, linewidth=2, facecolor='b')

#mt2 = [150, 87, 1]
mt2 = [90, 45, 90]
beachball(mt2, size=200, linewidth=2, facecolor='r')

mt3 = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
#mt3 = [-2.64209032e+24,  5.28142928e+23,  2.36085327e+24,  5.08171674e+22, -1.86496587e+24, -3.72439542e+23]
beachball(mt3, size=200, linewidth=2, facecolor='g')