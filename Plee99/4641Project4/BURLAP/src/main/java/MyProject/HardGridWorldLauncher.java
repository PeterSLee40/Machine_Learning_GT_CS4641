package MyProject;

//author: Peter Lee, plee99, 12/3/2018
//Inspired by: https://github.com/ckabuloglu/CS4641_HW4_MDPs_and_Reinforcement_Learning

import MyProject.util.AnalysisAggregator;
import MyProject.util.AnalysisRunnerHard;
import MyProject.util.HardRewardFunction;
import MyProject.util.HardTerminalFunction;
import MyProject.util.MapPrinter;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;

public class HardGridWorldLauncher {
	//These are some boolean variables that affect what will actually get executed
	private static boolean visualizeInitialGridWorld = true; //Loads a GUI with the agent, walls, and goal
	
	//runValueIteration, runPolicyIteration, and runQLearning indicate which algorithms will run in the experiment
	private static boolean runValueIteration = true; 
	private static boolean runPolicyIteration = true;
	private static boolean runQLearning = true;
	
	//showValueIterationPolicyMap, showPolicyIterationPolicyMap, and showQLearningPolicyMap will open a GUI
	//you can use to visualize the policy maps. Consider only having one variable set to true at a time
	//since the pop-up window does not indicate what algorithm was used to generate the map.
	private static boolean showValueIterationPolicyMap = true; 
	private static boolean showPolicyIterationPolicyMap = true;
	private static boolean showQLearningPolicyMap = true;
	
	private static Integer MAX_ITERATIONS = 100;
	private static Integer NUM_INTERVALS = 10;
	private static int B = -100;//lineBacker
	private static int W = 1; // wall
	private static int _ = 0; // nothing
	protected static int[][] userMap = new int[][] { 
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, B, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, B, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, B, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, B, _, _, _, _, B, _, _, _, _, B, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, B, _, B, _, B, _, B, _, B, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, W, _, W, _, W, _, W, _, W, _, _, _, _, _, _, _, _,},
		{ _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,},
		{ W, W, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, W, W,},
		{ W, W, W, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, W, W, W,},
		{ W, W, W, W, W, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, W, W, W, W, W,},
	};

//	private static Integer mapLen = map.length-1;

	public static void main(String[] args) {
		// convert to BURLAP indexing
		int[][] map = MapPrinter.mapToMatrix(userMap);
		int maxX = map.length-1;
		int maxY = map[0].length-1;
		// 

		HardGridWorld gen = new HardGridWorld(map,maxX,maxY); //0 index map is 11X11
		Domain domain = gen.generateDomain();

		State initialState = HardGridWorld.createStartState(	domain, 12, 0);
		System.out.println("Initial State:" + initialState.toString());
		
		
		double yardMultiplier = .5;
		double tackleMultiplier = .25;
		RewardFunction rf = new HardRewardFunction(maxX,maxY, map, yardMultiplier, tackleMultiplier); //Goal is at the top middle of grid
		
		
		
		TerminalFunction tf = new HardTerminalFunction(maxX,maxY, map); //Goal is at the top middle of grid
		
		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf,
				initialState);
		//Print the map that is being analyzed
		System.out.println("/////Hard Grid World Analysis/////\n");
		MapPrinter.printMap(MapPrinter.matrixToMap(map));
		
		if (visualizeInitialGridWorld) {
			visualizeInitialGridWorld(domain, gen, env);
		}
		
		AnalysisRunnerHard runner = new AnalysisRunnerHard(MAX_ITERATIONS,NUM_INTERVALS);
		if(runQLearning){
			runner.runQLearning(gen,domain,initialState, rf, tf, env, showQLearningPolicyMap);
		}
		if(runValueIteration){
			runner.runValueIteration(gen,domain,initialState, rf, tf, showValueIterationPolicyMap);
		}
		if(runPolicyIteration){
			runner.runPolicyIteration(gen,domain,initialState, rf, tf, showPolicyIterationPolicyMap);
		}
		
		
		AnalysisAggregator.printAggregateAnalysis();
	}



	private static void visualizeInitialGridWorld(Domain domain,
			HardGridWorld gen, SimulatedEnvironment env) {
		Visualizer v = gen.getVisualizer();
		VisualExplorer exp = new VisualExplorer(domain, env, v);

		exp.addKeyAction("w", HardGridWorld.ACTIONNORTH);
		exp.addKeyAction("s", HardGridWorld.ACTIONSOUTH);
		exp.addKeyAction("d", HardGridWorld.ACTIONEAST);
		exp.addKeyAction("a", HardGridWorld.ACTIONWEST);

		exp.setTitle("Hard Grid World");
		exp.initGUI();

	}
	

}
