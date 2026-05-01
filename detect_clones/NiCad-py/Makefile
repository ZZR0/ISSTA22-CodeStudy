# Recompile all NiCad tools and plugins

# Usage:
#    make [options]
#
#    where options can be one or more of:
#
#        MACHINESIZE=-mXX    - Force XX-bit tools, where XX=32 or 64 (default: system default)
#
#        SIZE=NNNN           - Compile TXL plugins to size -s NNNN (default: 400, max: 4000)
#                              (warning: larger sizes use more memory and slows syntax error recovery)
#				

all:
	cd tools; make clean; make; cd ..
	cd txl; make clean; make; cd ..

clean:
	cd tools; make clean; cd ..
	cd txl; make clean; cd ..
