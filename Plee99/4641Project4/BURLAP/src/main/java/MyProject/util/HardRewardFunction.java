package MyProject.util;

import MyProject.EasyGridWorld;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import java.lang.*;
//https://github.com/ckabuloglu/CS4641_HW4_MDPs_and_Reinforcement_Learning

public class HardRewardFunction implements RewardFunction {

	int goalX;
	int goalY;
	int[][] map;
	double tackleMultiplier;
	double yardMultiplier;
	public HardRewardFunction(int goalX, int goalY, int[][] map, double d, double e) {
		this.goalX = goalX;
		this.goalY = goalY;
		this.map = map;
		this.tackleMultiplier = d;
		this.yardMultiplier = e;
	}

	@Override
	public double reward(State s, GroundedAction a, State sprime) {
		// get location of agent in next state
		ObjectInstance agent = sprime.getFirstObjectOfClass(EasyGridWorld.CLASSAGENT);
		int ax = agent.getIntValForAttribute(EasyGridWorld.ATTX);
		int ay = agent.getIntValForAttribute(EasyGridWorld.ATTY);
		if (ay == this.goalY) {
			return 100;
		}
		//ay - 6 -  Math.abs(ax - this.goalX/2 )*.25
		// are they at goal location?
		return -1*(Math.abs(ax - this.goalX/2 )^2)*this.yardMultiplier - .1  + map[ax][ay]*this.tackleMultiplier;
	}

}
