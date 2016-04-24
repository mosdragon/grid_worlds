package myProj;

import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.vfa.fittedvi.FittedVI;
import burlap.behavior.singleagent.vfa.DifferentiableStateActionValue;
import burlap.behavior.singleagent.vfa.cmac.CMACFeatureDatabase;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.LunarLanderRF;
import burlap.domain.singleagent.lunarlander.LunarLanderTF;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
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
public class Lander {

  private static LunarLanderDomain lld = new LunarLanderDomain();
  private static Domain domain = lld.generateDomain();
  private static RewardFunction rf = new LunarLanderRF(domain);
  private static TerminalFunction tf = new LunarLanderTF(domain);

  private static State s = LunarLanderDomain.getCleanState(domain, 0);
  private static final int EPISODES = 5000;


  private static final double gamma = 0.99;
  private static final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
  //initial state generator
  private static final ConstantStateGenerator sg = new ConstantStateGenerator(s);

  /**
   * Q-Learning params
   */
  private static final double defaultQ = 0.5;
  private static final double qInit = 0.5;
  private static final double learning_rate = 0.1;




  public static void runVI() {

    double init_vals = 0.5;

    //setup vi with 0.99 discount factor, a value
    //function initialization that initializes all states to value 0.5, and which will
    //run for 30 iterations over the state space
    VITutorial vi = new VITutorial(domain, rf, tf, 0.99, new SimpleHashableStateFactory(),
        new ValueFunctionInitialization.ConstantValueFunctionInitialization(0.5), 30);

    //run planning from our initial state
    Policy p = vi.planFromState(s);

    //evaluate the policy with one roll out visualize the trajectory
    EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf);

    Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
    new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
  }

  public static void runQL() {
    LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

      public String getAgentName() {
        return "VI";
      }

      public LearningAgent generateAgent() {
        return new QLearning(domain, gamma, hashingFactory, qInit, learning_rate, 30);
      }
    };

    //define learning environment
    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, sg);

    //define experiment
    LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
        10, 100, qLearningFactory);

    exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
        PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
        PerformanceMetric.AVERAGEEPISODEREWARD);


    //start experiment
    exp.startExperiment();
  }


  public static void init() {
    LunarLanderDomain.setAgent(s, 0., 5.0, 0.0);
    LunarLanderDomain.setPad(s, 75., 95., 0., 10.);
  }

  public static void main(String[] args){

    init();
    runVI();


  }

  public static void visualize() {
    int nTilings = 5;
    CMACFeatureDatabase cmac = new CMACFeatureDatabase(nTilings,
        CMACFeatureDatabase.TilingArrangement.RANDOMJITTER);
    double resolution = 10.;

    double angleWidth = 2 * lld.getAngmax() / resolution;
    double xWidth = (lld.getXmax() - lld.getXmin()) / resolution;
    double yWidth = (lld.getYmax() - lld.getYmin()) / resolution;
    double velocityWidth = 2 * lld.getVmax() / resolution;

    cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
        domain.getAttribute(LunarLanderDomain.AATTNAME),
        angleWidth);
    cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
        domain.getAttribute(LunarLanderDomain.XATTNAME),
        xWidth);
    cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
        domain.getAttribute(LunarLanderDomain.YATTNAME),
        yWidth);
    cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
        domain.getAttribute(LunarLanderDomain.VXATTNAME),
        velocityWidth);
    cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
        domain.getAttribute(LunarLanderDomain.VYATTNAME),
        velocityWidth);

    DifferentiableStateActionValue vfa = cmac.generateVFA(defaultQ/nTilings);
    GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, 0.99, vfa, 0.02, 0.5);


//    Visualize
    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);
    List episodes = new ArrayList();
    for(int i = 0; i < 5000; i++){
      EpisodeAnalysis ea = agent.runLearningEpisode(env);
      episodes.add(ea);
      System.out.println(i + ": " + ea.maxTimeStep());
      env.resetEnvironment();
    }

    Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
    new EpisodeSequenceVisualizer(v, domain, episodes);
  }

}
