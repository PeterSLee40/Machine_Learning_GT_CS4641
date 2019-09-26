package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTestIters {
    /** The n value */
    private static int N = 100;
    
    public static void main(String[] args) {
        for(int trial = 0; trial < 5; trial++) {
        for (int iters = 10; iters < 100; iters += 10) {
        
        //for (N = 10; N < 200; N+=10) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iters);
        long starttime = System.currentTimeMillis();
        fit.train();
        System.out.print(iters + ",rhc," +ef.value(rhc.getOptimal()));
        System.out.print(", "+ (System.currentTimeMillis() - starttime));
        
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, iters);
        fit.train();
        System.out.print(",sa," + ef.value(sa.getOptimal()));
        System.out.print(", "+ (System.currentTimeMillis() - starttime));
        System.out.print(",");
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
        System.out.print(",");
        fit = new FixedIterationTrainer(ga, iters);
        System.out.print(",");
        fit.train();
        System.out.print(",");
        System.out.print(",ga," + ef.value(ga.getOptimal()));
        System.out.println(", "+ (System.currentTimeMillis() - starttime));
    }
    }
        
        //MIMIC mimic = new MIMIC(200, 5, pop);
        //fit = new FixedIterationTrainer(mimic, 1000);
        //fit.train();
        //System.out.println(",ga," + ef.value(mimic.getOptimal()));
    }
}
