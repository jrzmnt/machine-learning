## Reinforcement Learning: An Introduction

---

By very definition in reinforcement learning an agent takes action in the given environment either in continuous or discrete manner to maximize some notion of **reward** that is coded into it. Sounds too profound, well it is with a research base dating way back to classical behaviorist psychology, game theory, optimization algorithms etc. But, good for us lot of *‘setting the stage’* work has already been done for us to kick-start us directly into the problem formulation world and discover new things.

Essentially, most important of them all that reinforcement learning scenarios for an agent in deterministic environment can be formulated as dynamic programming problem. Fundamentally meaning agent has to perform series of steps in systematic manner so that it can learn the ideal solution and it will receive guidance from reward values. The equation that expresses such scenario in mathematical terms is known as **Bellman’s equation** which we will see in action in some time.

<p align="center">
<img src="https://miro.medium.com/max/510/1*CylzR3lBFqoMWMuJgjQo0w.png"/>
<br /><i>It takes into account some initial choices and future choices to come to formulate decision problem at certain point</i>
</p>

---

### Agent and Environment

Let’s first define concept of agent and environment formally before proceeding further for understanding technical details about RL. Environment is the universe of agent which changes state of agent with given action performed on it. Agent is the system that perceives environment via sensors and perform actions with actuators. In below situations Homer(Left) and Bart(right) are our agents and World is their environment. They performs actions on it and improve their state of being by getting happiness as reward.

<p align="center">
<img src="https://miro.medium.com/max/700/0*qkY7C93iOqAyUG1X.png"/>
<br /><i>Positive and Negative rewards increases or decreases tendency of that behavior. Eventually leading to better results in that environment over a period of time.</i>
</p>

---

### Recent Advancements and Scope

Starting with most popular game series since IBM’s Deep Blue v/s Kasparov which created huge hype and awareness for deep reinforcement learning is AlpaGo v/s Lee Sedol. Mastering a game with more board configuration than atoms in the Universe against a den 9 master shows the power such smart systems hold. Recent breakthroughs and wins against World Pros in creating Dota bots are also commendable OpenAI team, with bots getting trained to handle such complex and dynamic environment. Mastering these games are example of testing the limits of AI agent that can be created to handle very complex situations. Already complex applications like driver-less cars, smart drones are operating in real world. Let’s understand fundamentals of reinforcement learning and starts with OpenAI gym to make our own agent. After that move towards Deep RL and tackle more complex situations. Scope of its application is beyond imagination and can be applied to so many domains like time-series prediction, healthcare, supply-chain automation and so on.


> *The unique ability to run algorithm on same state over and over which helps it to learn best action for that state, which essentially is equivalent to breaking of construct of time for humans to gain infinite learning experience at almost no time.*

<p align="center">
<img src="https://miro.medium.com/max/1400/1*qpzAxoUR9POLYl__zJhU5g.png"/>
<br /><i>Policy Gradients with Monte Carlo Look Search Tree.
AlphaGo must restrict Breath and Depth of search among all board configurations with heuristics information supplied by training and winning policy for max reward.</i>
</p>

---

### Conceptual Understanding

With RL as a framework agent acts with certain actions which transforms state of the agent, each action is associated with reward value. It also uses a policy to determine its next action which maps states to action. A policy can be defined agent’s way of behaving at a given time. Now, policies can be deterministic and stochastic, finding an optimal policy is the key.

Also, **Different actions in different states will have different reward values**. Like *‘Fire’* command in a game of *Pocket Tanks* can’t always have same reward value associated with it, as sometimes it’s better to retain a position which is strategically good. To handle this complex dynamic problem with such huge combinations in a planned manner. We need Q-value (or action-value) table which stores a map of state-action pairs to rewards.

<p align="center">
<img src="https://miro.medium.com/max/643/0*4lnSi_R7yYI7uLBl.jpg"/>
<br /><i>Depending on terrain and position along with power combination being available our reward values will vary even after being present on same state.</i>
</p>

Now, defining **environment** in RL’s context as function, it takes action at a given state as input and returns new state and reward value associated with action-state pair.


<p align="center">
<img src="https://miro.medium.com/max/617/0*7PvoCPJuUAblMdcn.png"/>
<br /><i>For Games like Mario, Q-learning with CNN loss approximation can be used.</i>
</p>


Neural nets enter the picture with their ability to learn state-action pairs rewards with ease when the environment becomes complex and this is known as Deep RL. Like playing those earlier Atari games.

Here, we will limit to simple Q-Learning only w/o neural networks, where Q maps state-action pairs to a maximum with combination of immediate reward plus future rewards i.e. for new states learned value is reward plus future estimate of rewards. Quantifying it into a equation with different parameter like learning rate and discount factor to slow agent’s choice of action. We arrive onto following equation. Structurally, it holds similarity to Bellman’s equation.

<p align="center">
<img src="https://miro.medium.com/max/700/0*BPeyfQgVvGtB7E5U.png"/>
<br /><i>Q function Equation, tells about maximum expected cumulative award for given pair.</i>
</p>

---

## Hands On: Why OpenAI gym ?

> *A 2016 Nature survey indicated that more than 70 percent of researchers have tried and failed to reproduce another scientist’s experiments, and more than half have failed to reproduce their own experiments.*

OpenAI is created for removing this problem of lack of standardization in papers along with an aim to create better benchmarks by giving versatile numbers of environment with great ease of setting up. Aim of this tool is to increase reproducibility in the field of AI and provide tools with which everyone can learn about basics of AI.

---

*@source: [Towards Datascience Blog](https://towardsdatascience.com/)*