% TXL C 2018 Basis Grammar
% Version 6.0, July 2019

% Copyright 1994-2019 James R. Cordy, Andrew J. Malton and Christopher Dahn

% Simple null program to test the C grammar

% TXL C Grammar
include "c.grm"

% Ignore BOM headers from Windows
include "bom.grm"

% Just parse
function main
    replace [program] 
        P [program]
    by
	P
end function

