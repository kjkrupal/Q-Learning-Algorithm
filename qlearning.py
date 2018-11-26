import math
import random
import csv
import json
import numpy as np

def load_environment():
    environment = np.load('Environments/grid_game.npy')
    return environment

def initialize_q():
    Q = {}
    for i in range(10):
        for j in range(10):
            key1 = ((i+1, j+1), 'up')
            key2 = ((i+1, j+1), 'down')
            key3 = ((i+1, j+1), 'left')
            key4 = ((i+1, j+1), 'right')
            Q[key1] = 0
            Q[key2] = 0
            Q[key3] = 0
            Q[key4] = 0
    return Q

def get_possible_actions(current_state):
    
    global environment
    next_states = []
    up = (current_state[0] - 1, current_state[1])
    down = (current_state[0] + 1, current_state[1])
    left = (current_state[0], current_state[1] - 1)
    right = (current_state[0], current_state[1] + 1)
    
    if(not math.isnan(environment[up])):
        next_states.append('up')
    if(not math.isnan(environment[down])):
        next_states.append('down')
    if(not math.isnan(environment[left])):
        next_states.append('left')
    if(not math.isnan(environment[right])):
        next_states.append('right')
    
    return next_states

def get_boltzmann_probabilities(Q, state):

    global temperature
    global environment
    
    next_actions = get_possible_actions(state)
    action_probs = {}
    denominator = 0.0
    numerator = 0.0
    temperature = temperature - 0.03

    for action in next_actions:
        denominator += math.exp(Q[(state, action)] / temperature)
    
    for action in next_actions:
        probability = 0.0
        numerator = math.exp((Q[(state, action)])/temperature)
        if(denominator != 0):
            probability = numerator / denominator
        action_probs[action] = probability

    return action_probs

def epsilon_greedy_learning(current_state, Q, epsilon):
    
    global beta
    global alpha
    global environment
    global goal_state
    
    action_lookup = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    steps = 0
    
    
    while(current_state != goal_state):
        
        max_q_exploit = 0
        max_q = 0
        next_states = {}
        next_action = None
        possible_actions = get_possible_actions( current_state)
        
        for action in possible_actions:
            if (action == 'up'):
                next_states['up'] = (current_state[0] - 1, current_state[1])
            if (action == 'down'):
                next_states['down'] = (current_state[0] + 1, current_state[1])
            if (action == 'left'):
                next_states['left'] = (current_state[0], current_state[1] - 1)
            if (action == 'right'):
                next_states['right'] = (current_state[0], current_state[1] + 1)

        r = random.uniform(0, 1)
        
        # Exploit
        if(r < epsilon):
            for action in possible_actions:
                if(max_q_exploit <= Q[(current_state, action)]):
                    max_q_exploit = Q[(current_state, action)]
                    next_action = action
            
            if(next_action != None):
                next_state = next_states[next_action]
            else:
                next_action = possible_actions[(random.randrange(len(possible_actions)))]
                next_state = next_states[next_action]
                
        # Explore
        else:
            next_action = possible_actions[(random.randrange(len(possible_actions)))]
            next_state = next_states[next_action]
        
        next_possible_actions = get_possible_actions(next_state)
        
        for action in next_possible_actions:
            if (max_q <= Q[(current_state, action)]):
                max_q = Q[(current_state, action)]

        # Calculate Q
        Q[(current_state, next_action)] += alpha * (environment[current_state] + (beta * max_q) - Q[(current_state, next_action)])
        current_state = next_state
        steps += 1
    
    return(steps, Q)

def boltzmann_learning(current_state, Q):
    
    global beta
    global alpha
    global temperature
    global environment
    global goal_state
    
    action_lookup = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    steps = 0
    
    
    while(current_state != goal_state):
        
        max_q_exploit = 0
        max_q = 0
        next_states = {}

        possible_actions = get_possible_actions(current_state)
        
        for action in possible_actions:
            if (action == 'up'):
                next_states['up'] = (current_state[0] - 1, current_state[1])
            if (action == 'down'):
                next_states['down'] = (current_state[0] + 1, current_state[1])
            if (action == 'left'):
                next_states['left'] = (current_state[0], current_state[1] - 1)
            if (action == 'right'):
                next_states['right'] = (current_state[0], current_state[1] + 1)

        actions_prob = get_boltzmann_probabilities(Q, current_state)
        
        key_max = max(actions_prob.keys(), key=(lambda k: actions_prob[k]))
        key_min = min(actions_prob.keys(), key=(lambda k: actions_prob[k]))
        
        max_probability = actions_prob[key_max]
        min_probability = actions_prob[key_min]
        
        if ((max_probability - min_probability) <= 0.001):
            next_action = possible_actions[(random.randrange(len(possible_actions)))]
            next_state = next_states[next_action]
        
        else:
            next_action = key_max
            next_state = next_states[next_action]
        
        next_possible_actions = get_possible_actions(next_state)
        
        for action in next_possible_actions:
            if (max_q <= Q[(current_state, action)]):
                max_q = Q[(current_state, action)]

        # Calculate Q
        Q[(current_state, next_action)] += alpha * (environment[current_state] + (beta * max_q) - Q[(current_state, next_action)])
        current_state = next_state
        steps += 1
        
    return(steps, Q)

def convert_keys(mydict):
    for key in mydict.keys():
        if type(key) is not str:
            mydict[str(key)] = mydict[key]
            del mydict[key]
    return mydict

def generate_report(epsilon_values, Q_epsilons, epsilon_steps, Q_boltzmann, steps_boltzmann):
    
    global environment
    global goal_state
    global start_state
    
    Q_boltzmanns = convert_keys(Q_boltzmann)
    
    for epsilon in epsilon_values:
        Q_epsilons[epsilon] = convert_keys(Q_epsilons[epsilon])
    
    writeable_env = np.array2string(environment, precision=2, separator='  ', suppress_small=True)
    
    report = open('output.txt', 'w+')
    report.write('##### Q-Learning Algorithm #####\n\n')
    report.write('>  Grid Environment\n\n')
    report.write(writeable_env)
    report.write('\n\n')
    report.write('** Start state: ' + str(start_state) + '\n\n')
    report.write('** Goal state: ' + str(goal_state) + '\n\n')
    for epsilon in epsilon_values:
        report.write('>  Epsilon Greedy Q-values for epsilon = ' + str(epsilon) + '\n\n')
        report.write(' Iterations taken to reach convergence = ' + str(epsilon_steps[epsilon]) + '\n\n')
        report.write(' Q values:\n\n')
        report.write(json.dumps(Q_epsilons[epsilon], indent=4))
        report.write('\n\n')
    report.write('>  Boltzmann Q-values\n\n')
    report.write(' Iterations taken to reach convergence = ' + str(steps_boltzmann) + '\n\n')
    report.write(json.dumps(convert_keys(Q_boltzmann), indent=4))
    report.close()

def main():
    
    global beta
    global alpha
    global temperature
    global goal_state
    global environment
    global start_state
    
    beta = 0.9 
    alpha = 0.01
    temperature = 10
    start_state = (1, 1)
    goal_state = (6, 6)
    epsilon_values = [0.1, 0.2, 0.3]
    environment = load_environment()
    epsilon_steps = {}
    Q_epsilons = {}
    
    Q_epsilon = initialize_q()
    Q_boltzmann = initialize_q()
    
    
    for epsilon in epsilon_values:
        (steps_epsilon, Q_epsilon) = epsilon_greedy_learning(start_state, Q_epsilon, epsilon)
        Q_epsilons[epsilon] = Q_epsilon
        epsilon_steps[epsilon] = steps_epsilon
        
    (steps_boltzmann, Q_boltzmann) = boltzmann_learning(start_state, Q_boltzmann)
    generate_report(epsilon_values, Q_epsilons, epsilon_steps, Q_boltzmann, steps_boltzmann)

if __name__ == '__main__':
    main()

