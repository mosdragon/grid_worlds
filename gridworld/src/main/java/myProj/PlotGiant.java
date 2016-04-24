
package myProj;

import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.auxiliary.common.SinglePFTF;
import burlap.oomdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

import java.util.Random;

public class PlotGiant {

  static final double gamma = 0.8;
  static final double initvals = 0.5;
  static final double learning_rate = 0.2;

  static final double goal_reward = 400;
  static final double other_reward = -0.5;

  static final int dimX = 50;
  static final int dimY = 50;

  static final int goalX = 30;
  static final int goalY = 49;


  private static GridWorldDomain gw;
  private static Domain domain;
  //ends when the agent reaches a location
  private static TerminalFunction tf;

  //reward function definition
  private static RewardFunction rf;
  private static State initialState;


  private static Random random = new Random();

  private static final int TRIALS = 5;
  private static final int EPISODES = 1000;


  public static void init() {
    random.setSeed(0xABCDEF);

//    dimX x dimY grid world
    gw = new GridWorldDomain(dimX, dimY);

    gw.setProbSucceedTransitionDynamics(0.8); //stochastic transitions with 0.8 success rate

    final int num_walls = 300;
    int[] xwalls = new int[num_walls];
    int[] ywalls = new int[num_walls];

//    Choose x and y coords for walls
    for (int i = 0; i < xwalls.length && i < ywalls.length; i++) {
      int valx = random.nextInt(dimX);
      xwalls[i] = valx;

      int valy = random.nextInt(dimY);
      ywalls[i] = valy;
    }

//    Place the walls
    for (int i = 0; i < xwalls.length && i < ywalls.length; i++) {
      int xwall = xwalls[i];
      int ywall = ywalls[i];

//      Random wall length
      int walllen = random.nextInt(5);

//      Randomly determine if wall is vertical or horizontal
      if (random.nextBoolean()) {
        gw.horizontalWall(xwall, Math.min(xwall + walllen, dimX - 1), ywall);
      } else {
        gw.verticalWall(ywall, Math.min(ywall + walllen, dimY - 1), xwall);
      }
    }



    domain = gw.generateDomain(); //generate the grid world domain

    //setup initial state
    initialState = GridWorldDomain.getOneAgentOneLocationState(domain);

//    Begin agent at (0,0)
    GridWorldDomain.setAgent(initialState, 0, 0);

//    Set goal at (goalX, goalY)
    GridWorldDomain.setLocation(initialState, 0, goalX, goalY);

    //ends when the agent reaches a location
    tf = new SinglePFTF(domain.
        getPropFunction(GridWorldDomain.PFATLOCATION));

    //reward function definition
    rf = new GoalBasedRF(new TFGoalCondition(tf), goal_reward, other_reward);

  }

  private static void visualGame() {
    Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
    VisualExplorer exp = new VisualExplorer(domain, v, initialState);

    //set control keys to use w-s-a-d
    exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
    exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
    exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);
    exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);

    exp.initGUI();
  }

  private static void simulateQL() {
    //initial state generator
    final ConstantStateGenerator sg = new ConstantStateGenerator(initialState);

    //set up the state hashing system for looking up states
    final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

    /**
     * Create factory for Q-learning agent
     */
    LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

      public String getAgentName() {
        return "Q-learning";
      }

      public LearningAgent generateAgent() {
        return new QLearning(domain, gamma, hashingFactory, initvals, learning_rate);
      }
    };

    //define learning environment
    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, sg);

    //define experiment
    LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
        TRIALS, EPISODES, qLearningFactory);

    exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
        PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
        PerformanceMetric.AVERAGEEPISODEREWARD,
        PerformanceMetric.CUMULTAIVEREWARDPEREPISODE);


    //start experiment
    exp.startExperiment();


  }

  public static void main(String [] args) {

    init();
    visualGame();

  }
}