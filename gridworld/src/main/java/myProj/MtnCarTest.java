package myProj;

import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.vfa.fittedvi.FittedVI;
import burlap.behavior.singleagent.vfa.DifferentiableStateActionValue;
import burlap.behavior.singleagent.vfa.cmac.CMACFeatureDatabase;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.cartpole.InvertedPendulum;
import burlap.domain.singleagent.frostbite.FrostbiteDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.LunarLanderRF;
import burlap.domain.singleagent.lunarlander.LunarLanderTF;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;
import burlap.tutorials.cpl.VITutorial;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by osama on 4/23/16.
 */
public class MtnCarTest {

  private static final int EPISODES = 5000;


  private static final double gamma = 0.4;
  private static final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
  //initial state generator
//  private static final ConstantStateGenerator sg = new ConstantStateGenerator(s);

  /**
   * Q-Learning params
   */
  private static final double defaultQ = 0.5;
  private static final double qInit = 0.5;
  private static final double learning_rate = 0.1;


  static final double init_val = 0.0;
  static int iterations = 30;

  static BlockDude bdude;
  static Domain domain;
  static TerminalFunction tf;
  static RewardFunction rf;

  static double improvement_threshold = 1e-2;

  static int blocks = 5;


  public static void runPI() {


    Planner planner = new PolicyIteration(domain, rf, tf, gamma, hashingFactory,
        improvement_threshold, iterations, iterations);

    planner.toggleDebugPrinting(true);
//    Policy p = lspi.runPolicyIteration(iterations, 1e-6);
    State initialState = BlockDudeLevelConstructor.getLevel2(domain);

    Policy p = planner.planFromState(initialState);

    Visualizer v = BlockDudeVisualizer.getVisualizer(bdude.getMaxx(), bdude.getMaxy());
    VisualActionObserver vob = new VisualActionObserver(domain, v);
    vob.initGUI();

    State s = BlockDude.getUninitializedState(domain, blocks);
    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);

    env.addObservers(vob);

    for(int i = 0; i < 5; i++){
      p.evaluateBehavior(env);
      env.resetEnvironment();
    }

    System.out.println("Finished PI");

  }


  public static void runVI() {
//
////    FittedVI lspi = new FittedVI(domain, 0.99, fb, dataset);
//    VITutorial vi = new VITutorial(domain, rf, tf, gamma, hashingFactory, new
//        ValueFunctionInitialization.ConstantValueFunctionInitialization(init_val), iterations);
////    Policy p = lspi.runPolicyIteration(30, 1e-6);
//
//
//    Policy p = vi.planFromState(BlockDude.getCleanState(domain, bdude.physParams));

    State initialState = BlockDudeLevelConstructor.getLevel2(domain);

    Planner planner = new ValueIteration(domain, rf, tf, gamma, hashingFactory,
        improvement_threshold, iterations);
    Policy p = planner.planFromState(initialState);
//    p.evaluateBehavior(initialState, rf, tf);
//    .writeToFile(outputPath + "vi");

    System.out.println("Planned from state");
    p.evaluateBehavior(initialState, rf, tf).writeToFile("stuffvi.txt");

    Visualizer v = BlockDudeVisualizer.getVisualizer(bdude.getMaxx(), bdude.getMaxy());
    VisualActionObserver vob = new VisualActionObserver(domain, v);
    vob.initGUI();

    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, initialState);

    env.addObservers(vob);

    for(int i = 0; i < 5; i++){
      p.evaluateBehavior(env);
      env.resetEnvironment();
    }


    System.out.println("Finished VI");

  }

  public static void init() {
    bdude = new BlockDude();
    domain = bdude.generateDomain();
    tf = new BlockDudeTF();
    rf = new GoalBasedRF(tf, 100);

  }

  public static void main(String[] args){

    init();
    runPI();
//    runVI();

//    BlockDude.main(null);

  }

  public static void visualize() {
  }

}
