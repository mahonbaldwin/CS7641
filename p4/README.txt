# running p4

- checkout https://github.com/mahonbaldwin/CS7641/
- install python3
- download requirements from requirements.txt in p4
- from ~p4/gym-casino execute
  - pip install -e .

## Policy and Iteration Learning
To run the policy and iteration learning for bot the frozen lake and the roulette:

    cd analysis
    python3 mdp.py

## Frozen Lake Q-Learning

    cd frozen_lake
    python3 q_learning.py

Note: This will run for a LONG time to run the smaller examples run `python3 q_learning_20.py`

## Roulette Q-Learning

   cd roulette
   python3 q_learning.py