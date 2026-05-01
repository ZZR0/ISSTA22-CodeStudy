% NiCad abstract given nonterminals from potential clones - Java functions version
% Jim Cordy, May 2010

% Revised July 2018 - update to match new Java 8 grammar - JRC

% NiCad tag grammar
include "nicad.grm"

% Using Java grammar
include "java.grm"

define method_definition
    [method_header]
    '{  [NL][IN] 
	[repeat declaration_or_statement]
	[opt expression_statement_no_semi] [EX]
    '} 
end define

define method_header
	[repeat modifier] [opt generic_parameter] [opt type_specifier] [method_declarator] [opt throws] 
end define

define potential_clone
    [method_definition]
end define

% Generic nonterminal abstraction
include "generic-abstract.rul"
