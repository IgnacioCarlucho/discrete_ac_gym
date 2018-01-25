#Discret Actor-Critic

A discret actor-critic algorithm for solving the frozen lake problem. The environment is provided by gym. In the current example the slipery is disabled.
This is an On-Policy algorithm, with update rule: 

δ ← R + γ * vˆ(S,w) − vˆ(S,w)   
w ← w + alpha*δ ∇vˆ(S,w)    
θ ← θ + beta*δ∇lnπ(A|S, θ)   


##Requirements

- Tensorflow
- Gym
- Numpy 

##Running: 

python main.py

It runs the main algorithm. Both actor and critic have 2 hidden layers with 150 neurons. I used relu for the hidden layer and of course a softmax activation for the output layer. 


##Sources: 


- [Policy Gradient Methods, From David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)
- [Sutton](http://incompleteideas.net/book/bookdraft2018jan1.pdf)
- [DennyBritz RL examples](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient)