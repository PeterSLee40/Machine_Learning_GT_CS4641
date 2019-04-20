For this project I needed java 8, Ant, and ABIGAIL

the data generated from the graph was created using print statements located in the java files.
The printed Statements were typically copied and paste into excel, parsed, and the data extracted.
each test were hard-coded with appropriate parameters that were desired. The parameters are typically
at the top of the main method for each test.

The Dataset is stored in ABAGAIL-master/breastDataNoWords.csv
Many of the tests were run and their outputs were stored in ABAGAIL-master/Results

Files Created:
src.opt.test.geneticBreasts
src.opt.test.simulatedBreasts
src.opt.test.climbingBreasts
src.opt.test.TravelingSalesmanTest
src.opt.test.TravelingSalesmanIter
src.opt.test.CountOnesTestN
src.opt.test.FlipFlipTestIters
src.opt.test.FlipFlopTestedit

Files Edited:
src.opt.test.CountOnesTest
src.opt.example.NeuralNetworkEvaluationsFunction

In order to run tests, follow the instructions below.

PATH is the path to Abigail directory, in this project folder its "ABAGAIL-master"
TESTNAME is the desired test to be run, I have created a list below.
DESTINATION is the desired name of the .csv to export the data into.

cd PATH
ant
java -cp ABAGAIL.JAR opt.test.TESTNAME > DESTINATION.csv

all the testnames were written in ABAGAIL-master\src\opt\test

TESTNAME includes:

TravelingSalesmanTest
TravelingSalesmanIter
CountOnesTestN
CountOnesTest
NQueensTestit
geneticBreasts
climbingBreast
simulatedBreasts