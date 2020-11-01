from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import json
import itertools

# For Learning curve plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        self.discount = 1.0
        self.alpha = 0.1
        self.epsilon = 0.1
        self.loadQvalues()
        self.lastAction = 0
        self.moves = []
        self.lastState = 0
        self.gameCount = 0
        self.gameDoc = {}
        self.score = 0
        return

    def split(self, a, n):
            k, m = divmod(len(a), n)
            return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def createRanges(self):
        #Itializing ranges for qvalues and setting inital values [0,0,0]
        first = (list(self.split(range(-30,513), 15)))
        second = range(-20, 11)  
        third = (list(self.split(range(360), 15)))
        fourth = ( list(self.split(range(540), 15)))

        newList = [first, second, third, fourth]
        newList = list(itertools.product(*newList))


        return newList

    def loadQvalues(self):
        """
        Load q values 
        """
        self.qvalues = []
        self.qkeys = self.createRanges()
        try:
            with open('data/qvalues.json', 'r') as gameFile:
                self.qvalues = json.load(gameFile)
        except:
            for i in range(len(self.qkeys)):
                self.qvalues.append([0,0,0])

    def compareStates(self, state):
        for i in range(len(self.qkeys)):
            if state[0] in self.qkeys[i][0] and state[1] == self.qkeys[i][1] and state[2] in self.qkeys[i][2] and state[3] in self.qkeys[i][3]:
                state = self.qkeys[i]
                stateIndex = i
                break
        
        

        try: 
            return state, stateIndex
        except:
            print("shitty state shitty life: ")
            print(state)
            with open("data/gameDoc.json", "w") as gameFile:
                json.dump(self.gameDoc, gameFile)
            return stateIndex

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5}
    
    def observe(self, s1, a, r, s2, end,):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        if(end == False):
            self.qvalues[s1][2] += 1
            self.qvalues[s1][a] = self.qvalues[s1][a] + self.alpha * (r + self.discount  * max(self.qvalues[s2][0:2]) - self.qvalues[s1][a])
        else:
            self.qvalues[s1][2] += 1
            self.qvalues[s1][a] = self.qvalues[s1][a] + self.alpha * (r + self.discount * 0 - self.qvalues[s1][a])
            


        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # print("state: %s" % state)
        state, stateIndex = self.compareStates(state)


        if self.qvalues[stateIndex][0] >= self.qvalues[stateIndex][1]:
            action = 0
        else:
            action = 1

        return int(action), state, stateIndex

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
        
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        
        state = env.game.getGameState()
        formattedState = [int(state["player_y"]), int(state["player_vel"]), int(state["next_pipe_dist_to_player"]), int(state["next_pipe_top_y"])]
        
        
        # str(state["player_y"]) + "," + str(state["player_vel"]) + "," + str(state["next_pipe_dist_to_player"]) + "," + str(state["next_pipe_top_y"])
        action, correctState, stateIndex  = agent.training_policy(formattedState)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        # let the agent observe the current state transition

        getNewState = env.game.getGameState()
        
        newState = [int(getNewState["player_y"]), int(getNewState["player_vel"]), int(getNewState["next_pipe_dist_to_player"]), int(getNewState["next_pipe_top_y"])]
        newState, newStateIndex = agent.compareStates(newState)
        agent.observe(stateIndex, action, reward, newStateIndex, env.game_over())

        score += reward
        agent.score += reward
        # reset the environment if the game is overs
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            if score >= 50:
                saveScore = [score]
                with open('data/score.json', 'w') as scoreFile:
                    json.dump(saveScore, scoreFile)
            score = 0
            agent.lastState = 0


    if nb_episodes == 0:
        with open("data/qvalues.json", "w") as gameFile:
            json.dump(agent.qvalues, gameFile)





agent = FlappyAgent()
# train(5000, agent)
# episodes = [1000,1000,1000,1000,1000,1000]
episodes = [200,200,200,200,200,200]

x = [200,400,600,800,1000,1200]
# x = [1000,2000,3000,4000,5000,6000]
y = []


for episode in episodes:
    train(episode, agent)
    y.append(agent.score)




title = r"Learning Curve (Q-Learning,  $\alpha=0.1$,  $\epsilon=0.1$,  $\gamma=1.0$)"
yMinimum = max(y)
yMaximum = min(y)

plt.plot(x,y)
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title(title)


plt.show()