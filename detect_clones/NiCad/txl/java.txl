% TXL Java 8 Basis Grammar
% Version 4.2, October 2020

% Copyright 2001-2020 James R. Cordy, Xinping Guo and Thomas R. Dean

% Simple null program to test the Java grammar

% TXL Java 8 Grammar
include "java.grm"

% Ignore BOM headers 
include "bom.grm"

% Just parse
function main
    replace [program] 
        P [program]
    by
	P
end function

