import gym
import numpy as np

# 1. Load Environment and Q-table structure
env = gym.make('CartPole-v0')
Q = np.zeros([4,2])
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-leanring
eta = .628
gma = .9
epis = 3000
rev_list = [] # rewards per episode calculate
# 3. Q-learning Algorithm
for i in range(epis):
    # Reset environment
    s = env.reset()
    print(s)
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    
    while j < 99:
        print('{}%'.format((i/epis)))
        env.render()
        j+=1
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(0,1)*(1./(i+1)))
        print(a)
        #Get new state & reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rev_list.append(rAll)
    env.render()
    env.close()
print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)

# Reset environment
s = env.reset()
d = False
# The Q-Table learning algorithm
while d != True:
    env.render()
    # Choose action from Q table
    a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    #Get new state & reward from environment
    s1,r,d,_ = env.step(a)
    #Update Q-Table with new knowledge
    Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
    s = s1
# Code will stop at d == True, and render one state before it