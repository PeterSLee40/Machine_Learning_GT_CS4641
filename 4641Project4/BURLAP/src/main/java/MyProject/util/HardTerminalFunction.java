package MyProject.util;

import MyProject.EasyGridWorld;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;

public class HardTerminalFunction implements TerminalFunction {

	int goalX;
	int goalY;
	int[][] map;
	public HardTerminalFunction(int goalX, int goalY, int[][] map) {
		this.goalX = goalX;
		this.goalY = goalY;
		this.map = map;
	}

	@Override
	public boolean isTerminal(State s) {
		// get location of agent in next state
		ObjectInstance agent = s.getFirstObjectOfClass(EasyGridWorld.CLASSAGENT);
		int ay = agent.getIntValForAttribute(EasyGridWorld.ATTY);
		int ax = agent.getIntValForAttribute(EasyGridWorld.ATTX);
		// are they at goal location?
		if (ay == this.goalY || map[ax][ay] < 0) {
			return true;
		}
		return false;
	}

}
