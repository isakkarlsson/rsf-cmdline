This is a command line utility for running the random shapelet forest
on univariate time series in the file format found in the (UCR time
series repository)[http://www.cs.ucr.edu/~eamonn/time_series_data/].

To compile, you need Maven

    mvn clean install assemble:single

This will create a JAR-file in the `target` directory which can be run
using

    java -jar name-of-jar-file [OPTIONS] trainFile testFile

    usage: rsfcmd.jar [OPTIONS] trainFile testFile
    -l,--lower <arg>      Lower shapelet size (fraction of length, e.g, 0.05)
    -n,--no-trees <arg>   Number of trees
    -r,--sample <arg>     Number of shapelets
    -u,--upper <arg>      Upper shapelet size (fraction of length, e.g, 0.8)
	
	
A pre-compiled binary can be found in the `binary`-folder.
    
   
