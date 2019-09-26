package MyProject.util;

import MyProject.HardGridWorld;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;

import java.util.List;

public class AnalysisRunnerHard {

	final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

	private int MAX_ITERATIONS;
	private int NUM_INTERVALS;

	public AnalysisRunnerHard(int MAX_ITERATIONS, int NUM_INTERVALS){
		this.MAX_ITERATIONS = MAX_ITERATIONS;
		this.NUM_INTERVALS = NUM_INTERVALS;
		
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			AnalysisAggregator.addNumberOfIterations(numIterations);

		}

	}
	public void runValueIteration(HardGridWorld gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		System.out.println("//Value Iteration Analysis//");
		ValueIteration vi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.currentTimeMillis();
			vi = new ValueIteration(
					domain,
					rf,
					tf,
					0.99,
					hashingFactory,
					-1, numIterations); //Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations
										   //for comparison with the other algorithms.
	
			// run planning from our initial state
			p = vi.planFromState(initialState);
			AnalysisAggregator.addMillisecondsToFinishValueIteration((int) (System.currentTimeMillis()-startTime));

			// evaluate the policy with one roll out visualize the trajectory
			ea = p.evaluateBehavior(initialState, rf, tf);
			AnalysisAggregator.addValueIterationReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addStepsToFinishValueIteration(ea.numTimeSteps());
		}
		
//		Visualizer v = gen.getVisualizer();
//		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
		AnalysisAggregator.printValueIterationResults();
		MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
		System.out.println("\n\n");
		if(showPolicyMap){
			simpleValueFunctionVis((ValueFunction)vi, p, initialState, domain, hashingFactory, "Value Iteration");
		}
	}

	public void runPolicyIteration(HardGridWorld gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		System.out.println("//Policy Iteration Analysis//");
		PolicyIteration pi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.currentTimeMillis();
			pi = new PolicyIteration(
					domain,
					rf,
					tf,
					0.99,
					hashingFactory,
					-1, 1, numIterations);
	
			// run planning from our initial state
			p = pi.planFromState(initialState);
			AnalysisAggregator.addMillisecondsToFinishPolicyIteration((int) (System.currentTimeMillis() - startTime));

			// evaluate the policy with one roll out visualize the trajectory
			ea = p.evaluateBehavior(initialState, rf, tf);
			AnalysisAggregator.addPolicyIterationReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addStepsToFinishPolicyIteration(ea.numTimeSteps());
		}

//		Visualizer v = gen.getVisualizer();
//		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
		AnalysisAggregator.printPolicyIterationResults();

		MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
		System.out.println("\n\n");

		//visualize the value function and policy.
		if(showPolicyMap){
			simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration");
		}
	}

	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p, 
			State initialState, Domain domain, HashableStateFactory hashingFactory, String title){

		List<State> allStates = StateReachability.getReachableStates(initialState,
				(SADomain)domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
				allStates, valueFunction, p);
		gui.setTitle(title);
		gui.initGUI();

	}
	
	public void runQLearning(HardGridWorld gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf,
			SimulatedEnvironment env, boolean showPolicyMap) {
		System.out.println("//Q Learning Analysis//");
		/*
		 * Initializes Q-learning with 0.1 epsilon greedy policy, the same Q-value initialization everywhere, and places no limit on the number of steps the 
		 * agent can take in an episode. By default the agent will only save the last learning episode and a call to the {@link #planFromState(State)} method
		 * will cause the valueFunction to use only one episode for planning; this should probably be changed to a much larger value if you plan on using this
		 * algorithm as a planning algorithm.
		 * @param domain the domain in which to learn
		 * @param gamma the discount factor
		 * @param hashingFactory the state hashing factory to use for Q-lookups
		 * @param qInit the initial Q-value to user everywhere
		 * @param learningRate the learning rate
		public QLearning(Domain domain, double gamma, HashableStateFactory hashingFactory,
				double qInit, double learningRate)
		*/
		QLearning agent = null;
		agent = new QLearning(
					domain,
					0.99,
					hashingFactory,
					50, 0.5);
		Policy p = null;
		EpisodeAnalysis ea = null;
		long startTime = System.currentTimeMillis();
		int mult = 1000;
		int increment = MAX_ITERATIONS/NUM_INTERVALS*mult;
		for(int numIterations = 0; numIterations<=MAX_ITERATIONS*mult; numIterations++){
			ea = agent.runLearningEpisode(env);
			env.resetEnvironment();
			if (numIterations%increment == 0 && numIterations != 0) {
				agent.initializeForPlanning(rf, tf, 1);
				p = agent.planFromState(initialState);
				AnalysisAggregator.addQLearningReward(calcRewardInEpisode(ea));
				AnalysisAggregator.addMillisecondsToFinishQLearning((int) (System.currentTimeMillis() - startTime));
				AnalysisAggregator.addStepsToFinishQLearning(ea.numTimeSteps());
			}
		}
		/*
		for(int numIterations = increment*5; numIterations<=MAX_ITERATIONS*5; numIterations+=increment*5 ){
			startTime = System.nanoTime();
			agent.resetSolver();
			for (int i = 0; i < numIterations; i++) {
				ea = agent.runLearningEpisode(env);
				env.resetEnvironment();
			}
			agent.initializeForPlanning(rf, tf, 1);
			p = agent.planFromState(initialState);
			AnalysisAggregator.addQLearningReward(calcRewardInEpisode(ea));
			AnalysisAggregator.addMillisecondsToFinishQLearning((int) (System.nanoTime()-startTime)/1000000);
			AnalysisAggregator.addStepsToFinishQLearning(ea.numTimeSteps());

		}*/
		AnalysisAggregator.printQLearningResults();
		MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
		System.out.println("\n\n");

		//visualize the value function and policy.
		if(showPolicyMap){
			simpleValueFunctionVis((ValueFunction)agent, p, initialState, domain, hashingFactory, "Q-Learning");
		}

	}
	
	private static List<State> getAllStates(Domain domain,
			 RewardFunction rf, TerminalFunction tf,State initialState){
		ValueIteration vi = new ValueIteration(
				domain,
				rf,
				tf,
				0.95,
				new SimpleHashableStateFactory(),
				.5, 100);
		vi.planFromState(initialState);

		return vi.getAllStates();
	}
	
	public double calcRewardInEpisode(EpisodeAnalysis ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}
	
}
