% In this file, we have saved the output given by `PyBaMM <https://github.com/pybamm-team/PyBaMM>`_ for the data taken from Chen et al. paper.
% 
%      @article{Chen2020,
%      author = {Chen, Chang-Hui and Brosa Planella, Ferran and O'Regan, Kieran and Gastol, Dominika and Widanage, W. Dhammika and Kendrick, Emma},
%      title = {{Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models}},
%      journal = {Journal of The Electrochemical Society},
%      volume = {167},
%      number = {8},
%      pages = {080534},
%      year = {2020},
%      publisher = {The Electrochemical Society},
%      doi = {10.1149/1945-7111/ab9050},
%      }
% 
% The results are given by the arrays
%    t : time / h
%    u : Cell Voltage / V
%
% We also provide the values obtained for the case with (almost) instanteneous solid diffusion (solid diffusivity is set
% to 10 [m2.s-1]). The results are given in the arrays
%
%    t_infdiff : time / h
%    u_infdiff : Cell Voltage / V
%
% This is the code run in PyBaMM to obtain the results:
%
%
%         import numpy as np
%         import pybamm
%         
%         param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
%         
%         ## Uncomment two following lines for case with instanteneous solid diffusion
%         # param['Negative electrode diffusivity [m2.s-1]'] = 1e1
%         # param['Positive electrode diffusivity [m2.s-1]'] = 1e1
%         
%         CRate = 1
%         param['Current function [A]'] = 5 * CRate
%         
%         model = pybamm.lithium_ion.DFN()
%         sim = pybamm.Simulation(model, parameter_values=param)
%         
%         n = 100
%         t_eval = np.linspace(0, 3600 / CRate, n)
%         t_eval = np.concatenate([t_eval, [0.1]])
%         t_eval = np.sort(t_eval)
%         
%         sol = sim.solve(t_eval)
%         
%         u = sol['Terminal voltage [V]'].entries
%         t = sol['Time [h]'].entries


t = [0.        ,
     0.01010101,
     0.02020202,
     0.03030303,
     0.04040404,
     0.05050505,
     0.06060606,
     0.07070707,
     0.08080808,
     0.09090909,
     0.1010101 ,
     0.11111111,
     0.12121212,
     0.13131313,
     0.14141414,
     0.15151515,
     0.16161616,
     0.17171717,
     0.18181818,
     0.19191919,
     0.2020202 ,
     0.21212121,
     0.22222222,
     0.23232323,
     0.24242424,
     0.25252525,
     0.26262626,
     0.27272727,
     0.28282828,
     0.29292929,
     0.3030303 ,
     0.31313131,
     0.32323232,
     0.33333333,
     0.34343434,
     0.35353535,
     0.36363636,
     0.37373737,
     0.38383838,
     0.39393939,
     0.4040404 ,
     0.41414141,
     0.42424242,
     0.43434343,
     0.44444444,
     0.45454545,
     0.46464646,
     0.47474747,
     0.48484848,
     0.49494949,
     0.50505051,
     0.51515152,
     0.52525253,
     0.53535354,
     0.54545455,
     0.55555556,
     0.56565657,
     0.57575758,
     0.58585859,
     0.5959596 ,
     0.60606061,
     0.61616162,
     0.62626263,
     0.63636364,
     0.64646465,
     0.65656566,
     0.66666667,
     0.67676768,
     0.68686869,
     0.6969697 ,
     0.70707071,
     0.71717172,
     0.72727273,
     0.73737374,
     0.74747475,
     0.75757576,
     0.76767677,
     0.77777778,
     0.78787879,
     0.7979798 ,
     0.80808081,
     0.81818182,
     0.82828283,
     0.83838384,
     0.84848485,
     0.85858586,
     0.86868687,
     0.87878788,
     0.88888889,
     0.8989899 ,
     0.90909091,
     0.91919192,
     0.92929293,
     0.93939394,
     0.94949495,
     0.95959596,
     0.96969697,
     0.97979798,
     0.98698049];

u = [4.03250458,
     3.95109938,
     3.93503591,
     3.92942329,
     3.9253315 ,
     3.92029792,
     3.91348291,
     3.90492988,
     3.89510368,
     3.88452758,
     3.8736443 ,
     3.86277668,
     3.85214707,
     3.84188789,
     3.83206021,
     3.8226798 ,
     3.81371884,
     3.80511134,
     3.79675873,
     3.78853732,
     3.78032778,
     3.77204395,
     3.76363587,
     3.75507549,
     3.74634395,
     3.73742417,
     3.72829558,
     3.71893433,
     3.70931366,
     3.69940227,
     3.68917107,
     3.67859474,
     3.66765845,
     3.65637412,
     3.64481093,
     3.63312292,
     3.6215395 ,
     3.61028195,
     3.59948856,
     3.58921222,
     3.57945577,
     3.57020191,
     3.56142352,
     3.55308704,
     3.54515233,
     3.53757152,
     3.53029172,
     3.52325721,
     3.51641152,
     3.5097013 ,
     3.50307805,
     3.49649903,
     3.48992857,
     3.48333685,
     3.47669942,
     3.46999583,
     3.46320883,
     3.45632359,
     3.44932628,
     3.44220325,
     3.43494098,
     3.42752488,
     3.41993809,
     3.41216207,
     3.40417536,
     3.39595229,
     3.38746553,
     3.37868729,
     3.36959488,
     3.36018192,
     3.3504729 ,
     3.34053344,
     3.33046981,
     3.32040309,
     3.31042959,
     3.30058848,
     3.2908432 ,
     3.28109228,
     3.27119082,
     3.26097201,
     3.25025614,
     3.23883347,
     3.2264289 ,
     3.21263035,
     3.1967609 ,
     3.17771121,
     3.15401024,
     3.12484228,
     3.0913314 ,
     3.05593778,
     3.02060417,
     2.98555988,
     2.94924539,
     2.90874489,
     2.86003961,
     2.79791666,
     2.71558001,
     2.60392762,
     2.50000026];


t_infdiff = [0.0,
             0.010101010101010104,
             0.020202020202020207,
             0.030303030303030304,
             0.040404040404040414,
             0.05050505050505052,
             0.06060606060606061,
             0.0707070707070707,
             0.08080808080808083,
             0.09090909090909093,
             0.10101010101010104,
             0.11111111111111113,
             0.12121212121212122,
             0.13131313131313133,
             0.1414141414141414,
             0.15151515151515155,
             0.16161616161616166,
             0.17171717171717174,
             0.18181818181818185,
             0.19191919191919196,
             0.20202020202020207,
             0.21212121212121218,
             0.22222222222222227,
             0.23232323232323238,
             0.24242424242424243,
             0.25252525252525254,
             0.26262626262626265,
             0.27272727272727276,
             0.2828282828282828,
             0.292929292929293,
             0.3030303030303031,
             0.3131313131313132,
             0.3232323232323233,
             0.33333333333333337,
             0.3434343434343435,
             0.35353535353535354,
             0.3636363636363637,
             0.37373737373737376,
             0.3838383838383839,
             0.393939393939394,
             0.40404040404040414,
             0.4141414141414142,
             0.42424242424242437,
             0.4343434343434344,
             0.44444444444444453,
             0.45454545454545464,
             0.46464646464646475,
             0.4747474747474748,
             0.48484848484848486,
             0.49494949494949503,
             0.5050505050505051,
             0.5151515151515152,
             0.5252525252525253,
             0.5353535353535355,
             0.5454545454545455,
             0.5555555555555556,
             0.5656565656565656,
             0.5757575757575758,
             0.585858585858586,
             0.595959595959596,
             0.6060606060606062,
             0.6161616161616162,
             0.6262626262626264,
             0.6363636363636365,
             0.6464646464646466,
             0.6565656565656567,
             0.6666666666666667,
             0.6767676767676769,
             0.686868686868687,
             0.6969696969696971,
             0.7070707070707071,
             0.7171717171717172,
             0.7272727272727274,
             0.7373737373737376,
             0.7474747474747475,
             0.7575757575757577,
             0.7676767676767678,
             0.7777777777777779,
             0.787878787878788,
             0.7979797979797981,
             0.8080808080808083,
             0.8181818181818182,
             0.8282828282828284,
             0.8383838383838386,
             0.8484848484848487,
             0.8585858585858587,
             0.8686868686868688,
             0.878787878787879,
             0.8888888888888891,
             0.8989898989898991,
             0.9090909090909093,
             0.9191919191919193,
             0.9292929292929295,
             0.9393939393939394,
             0.9494949494949496,
             0.9595959595959598,
             0.9696969696969697,
             0.9797979797979799,
             0.9898989898989901,
             1.0];

u_infdiff = [4.03250457700196,
             4.0001198910591755,
             3.980650486154708,
             3.96651877357101,
             3.9558053588458675,
             3.947829433182422,
             3.942199108662444,
             3.938500992927514,
             3.9362057379393995,
             3.934689436637134,
             3.9333469966821104,
             3.9317101529385314,
             3.929472400391109,
             3.926440617648022,
             3.922491945168669,
             3.9175589520643768,
             3.911641235136881,
             3.904806409737552,
             3.8971730257956994,
             3.8888762695082737,
             3.88003347360852,
             3.870724949157772,
             3.8610136251049965,
             3.8509868214883163,
             3.8407645491301055,
             3.8304725243313245,
             3.8202143621953204,
             3.810064130783397,
             3.800069176482891,
             3.790251164510238,
             3.780614352960063,
             3.771149258857243,
             3.761829264272919,
             3.7526206221295606,
             3.7434791392592026,
             3.7343575222385863,
             3.7252214770841197,
             3.716081673558852,
             3.7070142092282845,
             3.6981301617280735,
             3.6894805287355545,
             3.6810068834579033,
             3.672582928423161,
             3.6640738710399927,
             3.655377067868355,
             3.6464295932624284,
             3.6372072529732318,
             3.6277202161892443,
             3.6180063272087866,
             3.608125000283114,
             3.598149442617691,
             3.5881577662100104,
             3.578225010761054,
             3.5684159043853874,
             3.558780063299207,
             3.549349274721505,
             3.5401368366033164,
             3.5311397387071644,
             3.5223417895848312,
             3.5137153556943836,
             3.505229665876663,
             3.4968466147364747,
             3.4885276848373477,
             3.4802350359033056,
             3.471930633805612,
             3.46357469950301,
             3.455133056445686,
             3.4465659833556774,
             3.437835330385534,
             3.428904976462236,
             3.4197420842013915,
             3.410324027568409,
             3.400650795976713,
             3.3907585860239675,
             3.3807266396696303,
             3.3706698953718464,
             3.360699604347568,
             3.3508750898270554,
             3.3411729781141037,
             3.3314815069406127,
             3.3216266044699103,
             3.3114031472164003,
             3.3005974067427792,
             3.2889855366476803,
             3.2762947278013543,
             3.2621461384791957,
             3.245945323526307,
             3.2267314664907047,
             3.2031274878298728,
             3.174086197797551,
             3.140659087072096,
             3.105652187181331,
             3.0709538502551577,
             3.036458995227646,
             3.0003585070791385,
             2.959634817299034,
             2.9102880013553545,
             2.8471477298180643,
             2.7633713664167807,
             2.6496738950219063];

