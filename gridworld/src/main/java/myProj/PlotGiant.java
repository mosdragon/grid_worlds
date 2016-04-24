
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

public class PlotGiant {

  public static void main(String [] args){

    GridWorldDomain gw = new GridWorldDomain(50,50); //11x11 grid world
//    gw.setMapToFourRooms(); //four rooms layout
    gw.setProbSucceedTransitionDynamics(0.8); //stochastic transitions with 0.8 success rate
    final Domain domain = gw.generateDomain(); //generate the grid world domain

    //setup initial state
    State s = GridWorldDomain.getOneAgentOneLocationState(domain);
    GridWorldDomain.setAgent(s, 0, 0);
    GridWorldDomain.setLocation(s, 0, 50, 50);



    //ends when the agent reaches a location
    final TerminalFunction tf = new GridWorldTerminalFunction(44, 45);

    //reward function definition
    final RewardFunction rf = new GoalBasedRF(tf, 20., -0.5);

    //initial state generator
    final ConstantStateGenerator sg = new ConstantStateGenerator(s);


    //set up the state hashing system for looking up states
    final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();


    //create visualizer and explorer
//    Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
//    VisualExplorer exp = new VisualExplorer(domain, v, s);
//
//    //set control keys to use w-s-a-d
//    exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
//    exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
//    exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);
//    exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);
//
//    exp.initGUI();


    /**
     * Create factory for Q-learning agent
     */
    LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

      public String getAgentName() {
        return "Q-learning";
      }

      public LearningAgent generateAgent() {
        return new QLearning(domain, 0.4, hashingFactory, 0.3, 0.1);
      }
    };

    //define learning environment
    SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, sg);

    //define experiment
    LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
        5, 1000, qLearningFactory);

    exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
        PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
        PerformanceMetric.AVERAGEEPISODEREWARD);


    //start experiment
    exp.startExperiment();


  }

}