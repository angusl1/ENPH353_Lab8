import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        with open(filename, 'rb') as file: 
            self.q = pickle.load(file)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename, 'wb') as file: 
            pickle.dump(self.q, file)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        # Perform a random action if the epsilon is higher than the random output
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            action = (action, self.getQ(state, action))
        else: 
            Qs = [(i, self.getQ(state,i)) for i in self.actions] # Compute Q value for all actions
            random.shuffle(Qs) # Shuffle Qs to avoid bias
            Qs.sort(key = lambda q: q[1]) # Sort Q values in ascending order
            action = Qs[-1] # Choose action with highest Q value

        return action if return_q else action[0]

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        q1 = self.q.get((state1, action1), None) # Get the Q value based on the state and action

        if q1 is None: 
            self.q[(state1, action1)] = reward # Initialize a Q value if there is none 
        else: 
            # Get the max Q value for the next state
            max_q2 = max([self.getQ(state2, i) for i in self.actions])
            # Update the max Q value for the state
            self.q[(state1, action1)] = q1 + self.alpha*(reward + self.gamma*max_q2 - q1)
