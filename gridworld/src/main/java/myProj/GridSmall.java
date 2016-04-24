
package myProj;

import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
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
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

import java.util.Random;

public class GridSmall {

  static final double gamma = 0.85;
  static final double initvals = 0.5;
  static final double learning_rate = 0.2;

  static final double goal_reward = 10;
  static final double other_reward = -0.5;

  static final int dimX = 11;
  static final int dimY = 11;

  static final int goalX = 10;
  static final int goalY = 10;


  private static GridWorldDomain gw;
  private static Domain domain;
  //ends when the agent reaches a location
  private static TerminalFunction tf;

  //reward function definition
  private static RewardFunction rf;
  private static State initialState;


  private static Random random = new Random();

  private static final int TRIALS = 5;
  private static final int EPISODES = 300;

  static final double improvement_threshold = 1e-2;


  static int valid_states = 0;


  //set up the state hashing system for looking up states
  static final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

  public static void init() {
    random.setSeed(0xABCDEF);

//    dimX x dimY grid world
    gw = new GridWorldDomain(dimX, dimY);
    gw.setMapToFourRooms(); //four rooms layout

    gw.setProbSucceedTransitionDynamics(0.8); //stochastic transitions with 0.8 success rate

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


//    valid_states = 0;
//    for (int[] row : gw.getMap()) {
//      for (int cell : row) {
//        if (cell == 0) {
//          valid_states++;
//        }
//      }
//    }
//    System.out.println("Total possible states: " + valid_states);

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

    long start = System.nanoTime();

    //initial state generator
    final ConstantStateGenerator sg = new ConstantStateGenerator(initialState);

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

    long end = System.nanoTime();
    long deltaTime = end - start;
    double seconds = deltaTime / (double) (1E9);
    System.out.println("================================================");
    System.out.println("Finished QL: " + seconds + " s.");
    System.out.println("================================================");


  }


  private static void simulateVI() {


    long start = System.nanoTime();

    Planner planner = new ValueIteration(domain, rf, tf, gamma, hashingFactory,
        improvement_threshold, EPISODES);
    Policy p = planner.planFromState(initialState);

    Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
    VisualActionObserver vob = new VisualActionObserver(domain, v);
    vob.initGUI();

    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, initialState);

    env.addObservers(vob);

    for(int i = 0; i < 5; i++){
      p.evaluateBehavior(env);
      env.resetEnvironment();
      System.out.println("Done with iteration: " + i+1);
    }

    long end = System.nanoTime();
    long deltaTime = end - start;
    double seconds = deltaTime / (double) (1E9);
    System.out.println("================================================");
    System.out.println("Finished VI: " + seconds  + " s.");
    System.out.println("================================================");

  }

  private static void simulatePI() {

    long start = System.nanoTime();

    Planner planner;
    planner = new PolicyIteration(domain, rf, tf, gamma, hashingFactory, improvement_threshold,
        EPISODES, EPISODES);
    Policy p = planner.planFromState(initialState);

    Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
    VisualActionObserver vob = new VisualActionObserver(domain, v);
    vob.initGUI();

    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, initialState);

    env.addObservers(vob);

    for(int i = 0; i < 5; i++){
      p.evaluateBehavior(env);
      env.resetEnvironment();
      System.out.println("Done with iteration: " + i + 1);
    }

    long end = System.nanoTime();
    long deltaTime = end - start;
    double seconds = deltaTime / (double) (1E9);
    System.out.println("================================================");
    System.out.println("Finished PI: " + seconds + " s.");
    System.out.println("================================================");

  }

  public static void main(String [] args) {

    init();
    visualGame();
//    simulateQL();
    simulateVI();
    simulatePI();
    simulateQL();
  }
}