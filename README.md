# Running the agent

## Using docker

Run commands from the root of the repository:
```
docker build -t frozen_lake .
docker run -it -v $(PWD)/output:/python/output frozen_lake
```

## Using virtualenv

Create virtualenv with `python3.9`, activate it and run from the root of the repository:
```
pip install -r requriements.txt
pip install -e .
python solution/train_agent.py
```

# Agent

* Used algorithm: QLearning.
* Used step size decay to speed up learning in early phases and to achieve more stable results at the end of training.
* Explortation achieved with optimistic initialization.
* 

# Results

## Rewards versus number of episodes
[rewards.png](output/rewards.png)

## Learned agent vizualization 
[agent.png](output/agent.png)

