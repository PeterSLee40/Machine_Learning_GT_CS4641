package opt.test;

import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.tester.AccuracyTestMetric;
import shared.tester.ConfusionMatrixTestMetric;
import shared.tester.NeuralNetworkTester;
import shared.tester.TestMetric;
import shared.tester.Tester;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import shared.Instance;
import shared.reader.ArffDataSetReader;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.lang.String;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.File;

/**
 * @author Peter Lee
 * @date 2018-10-30
 */
public class simulatedBreasts {

    /**
     * Create a neural net to classify the edibility of mushrooms
     * @param args ignored
     */
    public static void main(String[] args) {
        String[][] a;
        Instance[] data;
        // Create the validation and training sets from the csv file
        try{
            a = readCSV();
            //System.out.println(a[0][0]);
            data = initializeInstances(a);
            //System.out.println("initialized instances");
            DataSet set = new DataSet( data);
            int inputLayer = data[0].getData().size() - 1;

            //System.out.println(sets[0]);
            FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
            // There are 22 inputs then hidden layers of 15 and 10
            FeedForwardNetwork network = factory.createClassificationNetwork(new int[] {inputLayer, 10, 5, 1 });
            ErrorMeasure measure = new SumOfSquaresError();

            NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
                set, network, measure);
            //System.out.println("Randomized Hill Climbing");
            //System.out.println("========================");
            // We will optimize the neural net with randomized hill climbing
            
            SimulatedAnnealing o = new SimulatedAnnealing(100, .50, nno);
            FixedIterationTrainer fit = new FixedIterationTrainer(o, 5000);

            // Fit the data to find the weights using the optimization algorithm
            long startTime = System.currentTimeMillis();
            fit.train();
            long endTime = System.currentTimeMillis();
            long timeElapsed = (endTime- startTime);
            //System.out.println("Time Elapsed " + timeElapsed / 1000.0 + " seconds");
            // Pick the best weights that we found and set the neural net weights
            Instance opt = o.getOptimal();
            network.setWeights(opt.getData());

        } catch(Exception e) {System.out.println("fail");}
        

        

        // Print the results
        //System.out.println(network.value(data[0]));
        //DataSetUtils.printStatistics(set, network, false);

    }

    public static String[][] readCSV() throws FileNotFoundException, IOException {
        String[][] lol = new String[568][31];
        FileReader file = new FileReader("breastDataNoWords.csv");
        BufferedReader br = new BufferedReader(file);
        String line = br.readLine();
        int i =0;
        int j = 0;
        while ((line = br.readLine()) != null && !line.isEmpty()) {
            List<String> fields = Arrays.asList(line.split(","));
            i++;
            j = 0;
            for (String s: fields) {
                lol[i][j] = s;
                j++;
            }
        }
        br.close();

    return lol;
}


    private static Instance[] initializeInstances(String[][] arr) {
        Instance[] inarr = new Instance[567];
        for (int q = 0; q < 567; q++) {
            double[] data = new double[30];
            boolean label;
            for (int u = 0; u < 30; u++) {
                data[u] = Double.parseDouble( arr[q + 1][u] );
            }
            label = (Integer.valueOf(arr[q + 1][30]) == 1);
            Instance newIn = new Instance(data, label);
            inarr[q] = newIn;
        }
        return inarr;
    }
}
