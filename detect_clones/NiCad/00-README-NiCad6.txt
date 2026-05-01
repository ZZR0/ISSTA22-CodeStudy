NiCad clone detection system, Version 6.1 (15 July 2020)
--------------------------------------------------------
Software Technology Laboratory, Queen's University
April 2010 (Revised July 2020)

Copyright 2011-2020, J.R. Cordy & C.K. Roy

This directory contains all of the parsers and tools necessary to
install and run the NiCad near-miss clone detection system.
NiCad should compile and run correctly on all Linux, Mac OS X and 
Cygwin systems, but does not at present run directly under Windows. 

Installing and Running NiCad
----------------------------
NiCad can be installed on Ubuntu, Mac OS X, Cygwin, MinGW and other 
Unix-like systems with a GCC compiler and a FreeTXL distribution.

1. NiCad 6.1 requires that FreeTXL 10.8 or later be installed on your system.  
   FreeTXL can be downloaded from: 

      http://www.txl.ca/

   Install TXL 10.8 or later before proceeding.

2. NiCad optimizes by using precompiled TXL programs.  Use the command:

      make

   in this directory to precompile all of the NiCad tools and TXL plugins 
   before using NiCad. 

3. (Installation-free option) NiCad requires no further installation for 
   personal use. If you are not planning to install NiCad globally for 
   everyone on your system, and just want to use it yourself in this 
   directory, you can stop here and move on to "Testing NiCad" below.

4. (Optional global installation option) To install NiCad permanently for 
   everyone on the computer (for example, if you have a research group 
   all working on the same compute server), 

     (i) Copy the entire contents of this directory to any place where 
         it can reside permanently. On most systems the appropriate 
         place would be /usr/local/lib/nicad6.

              sudo mkdir /usr/local/lib/nicad6
              sudo cp -r . /usr/local/lib/nicad6

     (ii) Edit the command scripts "nicad6" and "nicad6cross" in this 
         directory to specify the directory where you installed NiCad, 
         and the kind of your system.  For example, if you put NiCad in
         /usr/local/lib/nicad6, you would modify the scripts to say:

              LIB=/usr/local/lib/nicad6

      (iii) Copy the edited "nicad6" and "nicad6cross" scripts to any "bin" 
         directory which is on the command search path of the intended 
         users of NiCad.  On most systems the appropriate place would be 
         /usr/local/bin.

              sudo cp nicad6 nicad6cross /usr/local/bin

Using NiCad
-----------
To run NiCad on a new system,

1. In a command line window, chage directory to any writable directory
   where you intend to run NiCad. If you are using the installation-free 
   option and running NiCad in place, change to this directory.

      cd NiCad-6.1 

2. Make an analysis directory to hold the source systems and results of NiCad 
   clone analysis.  If you are using the installation-free option and running 
   NiCad in place, then ./systems in the NiCad distribution directory is the 
   appropriate place.

      mkdir ./systems

3. Copy or symbolically link the entire source directory of the system you want 
   to analyze to the analysis directory (e.g., cp -r ./examples/JHotDraw ./systems/). 
   You can use a symbolic link (e.g., ln -s ./examples/JHotDraw ./systems/)
   to avoid large copies if you wish, but you should be aware that NiCad may
   write temporary analysis files into the system's source directory as part of
   the analysis process.

   At present NiCad comes with plugins to handle systems with source files 
   written in the languages: C (.c), C# (.cs), Java (.java), Python (.py), 
   PHP (.php), Ruby (.rb), WSDL (.wsdl) and ATL (.atl). 
   (TXL plugins for other languages can easily be added.)

4. Run the NiCad command on it, specifying the analysis granularity and language
   of the system you want to analyze, e.g.,

      ./nicad6 functions java systems/JHotDraw default-report

   At present NiCad can handle the granularities: functions and blocks.

5. Examine the results in the system's clone results directory in your analysis 
   directory, e.g., systems/JHotDraw_functions--blind-clones for the command above.
   NiCad results are reported in three ways: as clone pairs in XML-like format,
   as clone classes in XML-like format (both with and without original sources), 
   and as browsable HTML pages with clone classes and original source for each clone.

   For the command above, the following results files will be created
   in that directory:

      Clone pairs in XML-like format:
          JHotDraw_functions-blind-clones-0.30.xml

      Clone classes (clusters) in XML-like format:
          JHotDraw_functions-blind-clones-0.30-classes.xml

   If reports are specified in the configuration (as in "default-report" above)
   then the following additional results files will be created:

      Clone classes (clusters) with original sources for clones in XML-like format:
          JHotDraw_functions-blind-clones-0.30-classes-withsource.xml

      Clone classes (clusters) with original sources for clones as an HTML web-page report:
          JHotDraw_functions-blind-clones-0.30-classes-withsource.html

      The HTML web-page report can be opened and viewed in any standard web browser.

   The 0.30 (or equivalently, 0.00, 0.10, 0.20, ...) indicates the near-miss 
   difference threshold used by NiCad in the clone detection run, where 0.00 means 
   exact clones, 0.10 means at most 10% different pretty-printed lines, and so on.
   The default near-miss threshold is 0.30, as shown above.

6. NiCad supports a wide range of customized clone detection options including
   renaming, filtering, abstraction and custom normalization before comparison 
   using configuration files stored in the ./config subdirectory of the NiCad 
   installation directory.  To use a configuration, run NiCad giving the name of 
   the configuration as the last parameter on the command line.  

   E.g., to use the consistent renaming configuration with HTML web-page reporting,

      ./nicad6 functions java systems/JHotDraw rename-consistent-report

   When using different configurations, the requested transformations will be 
   applied and the results reported in different directory, e.g., 

      JHotDraw_functions-clones
      JHotDraw_functions-blind-clones
      JHotDraw_functions-consistent-clones
      JHotDraw_functions-blind-abstract-clones

   and so on.

   The default configurations ("default" and "default-report"), specify blind
   renaming with a near-miss threshold of 0.30, to aggressively find Type 3-2 clones.

7. To re-run NiCad on a system, for example with a different configuration,
   you can simply run the NiCad command on the system again:

       ./nicad6 functions java systems/JHotDraw type2-report

   NiCad will optimize to avoid re-running extraction for the same granularity,
   so subsequent runs on the same system will be significantly faster.

   To remove all clone detection results and intermediate file to start over 
   and force a new extraction from the same system, run the command:

       ./cleanall systems/JHotDraw 

   To remove all results and intermediate files of all previous NiCad runs,
   use the command:
     
       ./cleanall 

NiCadCross
-----------
NiCadCross is the NiCad cross-clone detector.  It does an cross-system test 
- that is, given two systems s1 and s2, it reports only clones of fragments 
of s1 in s2.  This is useful in incremental clone detection for new versions, 
or for detecting clones between two systems to be checked for cross-cloning.

NiCadCross is run in much the same way as NiCad, but giving a second system
source directory on the command line, for example:

      ./nicad6cross functions java systems/JHotDraw54b1 systems/JHotDraw76a2 default-report

Results are stored in the first system's cross-clone results directory, 
e.g., systems/JHotDraw541_functions-blind-crossclones for the command above.

Maintenance and Extension of NiCad
-----------------------------------
Maintaining or adding NiCad TXL plugins is easy - you just create the new
programs with appropriate names (see the ./txl subdirectory of the NiCad 
installation directory for examples), and NiCad will automatically allow
your new plugins to be used as normalizations or languages.

If you plan to change, maintain or recompile the NiCad clone comparison
tools themselves, you will require Turing Plus 6.0 (2018) or later to be 
installed on your system.  See the ./tools subdirectory of the NiCad 
installation for details.  

Turing Plus can be downloaded from: 
 
      http://txl.ca/tplus/

JRC 15.7.20
