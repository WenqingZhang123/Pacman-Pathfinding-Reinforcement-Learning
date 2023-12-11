# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # low noise will make the agent more likely to choose the optimal action and reach the goal state
    answerDiscount = 0.9
    answerNoise = 0.002
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.91
    answerNoise = 0.09
    # the living reward should be negative to encourage the agent to reach a terminal state
    answerLivingReward = -3.5
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    # much smaller than 3a to encourage the agent to reach the terminal state
    answerDiscount = 0.09
    answerNoise = 0.09
    # much bigger than  3a not to reach the cliff
    answerLivingReward = - 2.0
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    answerDiscount = 0.99
    answerNoise = 0.09
    # much smaller than 3a
    answerLivingReward = - 1.0
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    # More future reward to encourage the agent to reach the far exit
    answerDiscount = 0.99
    answerNoise = 0.18
    # make it explore more
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    answerDiscount = 0.99
    answerNoise = 0.2
    # it will keep exploring with a big positive living reward
    answerLivingReward = 10.0
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    return 'NOT POSSIBLE'
    answerEpsilon = 0.05
    answerLearningRate = 0.7
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
