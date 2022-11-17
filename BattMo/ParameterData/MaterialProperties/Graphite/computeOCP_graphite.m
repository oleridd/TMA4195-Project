function [OCP, dUdT] = computeOCP_graphite(c, T, cmax)

    Tref = 298.15;  % [K]
    
    theta = c./cmax;
    
    % Calculate the open-circuit potential at the reference temperature for the given lithiation
    refOCP = (0.7222 ...
              + 0.1387 .* theta ...
              + 0.0290 .* theta.^0.5 ...
              - 0.0172 ./ theta ... 
              + 0.0019 ./ theta.^1.5 ...
              + 0.2808 .* exp(0.9 - 15.*theta) ... 
              - 0.7984 .* exp(0.4465.*theta - 0.4108));
    
    coeff1 = [0.005269056 ,...
              + 3.299265709,...
              - 91.79325798,...
              + 1004.911008,...
              - 5812.278127,...
              + 19329.75490,...
              - 37147.89470,...
              + 38379.18127,...
              - 16515.05308];
    
    coeff2= [1, ...
             - 48.09287227,...
             + 1017.234804,...
             - 10481.80419,...
             + 59431.30000,...
             - 195881.6488,...
             + 374577.3152,...
             - 385821.1607,...
             + 165705.8597];
    
    dUdT = 1e-3.*polyval(coeff1(end:-1:1),theta)./ polyval(coeff2(end:-1:1),theta);

    % Calculate the open-circuit potential of the active material
    OCP = refOCP + (T - Tref) .* dUdT;
    
end


%{
Copyright 2021-2022 SINTEF Industry, Sustainable Energy Technology
and SINTEF Digital, Mathematics & Cybernetics.

This file is part of The Battery Modeling Toolbox BattMo

BattMo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BattMo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BattMo.  If not, see <http://www.gnu.org/licenses/>.
%}
