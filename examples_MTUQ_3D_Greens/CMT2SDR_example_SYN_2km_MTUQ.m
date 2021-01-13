% close all
M0 = zeros(3);
% %srf-event:
% strike = 90 	 	 ; dip = 45; rake = 90; %source0 	20110429190800 	-43.1842 	171.9868 	69 	90 	151 	159 	61 	0
strike =  	263 	 ; dip = 84; rake =  -163; %source0 	2013p544960 	171 	73 	-7
% %Cartesian format: east/ north/ depth or x/ y/ z
MT = sdr2mt(strike,dip,rake);
%M0 = [Mrr, Mrp, Mrt; Mrp, Mpp, Mpt; Mrt, Mpt, Mtt]; %Similar to obspy format!
M0(1,1) = MT(1); M0(3,3) = MT(2); M0(2,2) = MT(3);
M0(1,2) = MT(5); M0(1,3) = MT(4); M0(2,3) = MT(6);
M0(2,1) = MT(5); M0(3,1) = MT(4); M0(3,2) = MT(6);
disp(M0)
fig = figure(1);
set(gcf,'Position',[100 100 600 150])
subplot(1,2,1)
plotmt(M0,0,0,1,'k'), axis equal
title(num2str([strike,dip,rake]))
% M0 = [0,0,0;0,0,0;0,0,1]
%SYN
% % M6 = [1.24572585e+23 -1.26087676e+23  8.10047682e+21  1.55759184e+23  1.13297217e+23 -5.56568079e+22];
% M0 = [-0.02567095  0.30070382 -0.18111702;  0.30070382  0.09517079 -0.44115079;  -0.18111702 -0.44115079 -0.0674272];%bw
M0 = [-40.13031343  21.9826507   -3.17294287;  21.9826507   23.33717592   -0.04111206;  -3.17294287  -0.04111206  16.51479718];%sw
% M0 = [-0.01435231  0.08032913 -0.21891715;  0.08032913  0.05193741 -0.06526979;  -0.21891715 -0.06526979 -0.03514885];%bw M10
% M0 = [-45.61144923  23.91311786  -2.65217459;  23.91311786  11.66807091    0.54430658;  -2.65217459   0.54430658  33.65204103];%sw M10

% M0 = [-502.02766794  -67.86015827  221.38680986;  -67.86015827  580.86777986  -263.03997597;  221.38680986 -263.03997597  -77.58318633];%SYN M10
% M0 = [-365.78489061  -82.36698103   91.3057977;   -82.36698103  114.24945316     3.45988882;   91.3057977     3.45988882  251.05018409];%SYN M00

% M6 = [-1.83993602e+24 -7.17073380e+23  2.05172854e+24 -1.72736349e+23 -1.17998048e+24  5.35363906e+23];
%Plot Solution
% figure(10)

% subplot(1,2,2)
% plotmt(M0,0,0,1,'k'), axis equal
% convert strike-dip-rake to HRV mt
%[Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]
% Mrr = M0(3,3); Mtt = M0(1,1); Mpp = M0(2,2);
% Mrt = M0(1,3); Mrp = -M0(2,3); Mtp = -M0(1,2);

% Mrr =  M6(3);Mtt =  M6(1); Mpp =  M6(2); Mrt =  M6(5); Mrp = -M6(6); Mtp = -M6(4);



%Plot Solution
% figure(10)

subplot(1,2,2)
plotmt(M0,0,0,1,'k'), axis equal
% M0=M0i;
% % convert strike-dip-rake to HRV mt
% %[Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]
Mrr = M0(3,3); Mtt = M0(1,1); Mpp = M0(2,2);
Mrt = M0(1,3); Mrp = -M0(2,3); Mtp = -M0(1,2);
% % f = strike, d = dip, l = rake
% %1 Mrr =  Mzz =  Mo sin2d sinl
% %2 Mtt =  Mxx = -Mo(sind cosl sin2f +     sin2d sinl (sinf)^2 )
% %3 Mpp =  Myy =  Mo(sind cosl sin2f -     sin2d sinl (cosf)^2 )
% %4 Mrt =  Mxz = -Mo(cosd cosl cosf  +     cos2d sinl sinf )
% %5 Mrp = -Myz =  Mo(cosd cosl sinf  -     cos2d sinl cosf )
% %6 Mtp = -Mxy = -Mo(sind cosl cos2f + 0.5 sin2d sinl sin2f )
% %Havard format: up/ south, east or radius/ theta/ phi
sdr = round(mt2sdr([Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]));
%Correct the strike rotation!
sdr(1) = sdr(1)+90
% sdr = round(mt2sdr([M6(1),M6(2),M6(3),M6(4),M6(5),M6(6)]));
% sdr = [75,30,105]
title(num2str(sdr))
% saveas(fig,'Balls.jpg');
% savefig(fig,'Balls.jpg');

MT = sdr2mt(sdr);
%M0 = [Mrr, Mrp, Mrt; Mrp, Mpp, Mpt; Mrt, Mpt, Mtt]; %Similar to obspy format!
M0(1,1) = MT(1); M0(3,3) = MT(2); M0(2,2) = MT(3);
M0(1,2) = MT(5); M0(1,3) = MT(4); M0(2,3) = MT(6);
M0(2,1) = MT(5); M0(3,1) = MT(4); M0(3,2) = MT(6);

plotmt(M0,0,0,1,'k'), axis equal
title(num2str(sdr))
% title('[Mxx, Myy, Mzz, Mxy, Mxz, Myz] = [0.37942488, -0.51724356,  0.13781868,  0.73431469, -0.49435213,        0.03389469]')
% title('[Mxx, Myy, Mzz, Mxy, Mxz, Myz] = [-0.51724356,  0.37942488,  0.13781868, -0.73431469,  0.03389469,        0.49435213]')
% title(strcat('[Mxx, Myy, Mzz, Mxy, Mxz, Myz] = ',num2str([Mtt, Mpp, Mrr, -Mtp, Mrt, -Mrp])))
saveas(fig,'Geonet_vs_direct_CMT_solution.jpg');