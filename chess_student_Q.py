import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *

size_board = 4

# set a random seed so results can be reproduced
#np.random.seed(9)
np.random.seed(10)
#np.random.seed(11)

def main():
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of 
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1): 
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights  
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)

    """
    Initialization
    Define the size of the layers and initialization
    FILL THE CODE
    Define the network, the number of the nodes of the hidden layer should be 200, you should know the rest. The weights 
    should be initialised according to a uniform distribution and rescaled by the total number of connections between 
    the considered two layers. For instance, if you are initializing the weights between the input layer and the hidden 
    layer each weight should be divided by (n_input_layer x n_hidden_layer), where n_input_layer and n_hidden_layer 
    refer to the number of nodes in the input layer and the number of nodes in the hidden layer respectively. The biases
     should be initialized with zeros.
    """
    n_input_layer = 50  # Number of neurons of the input layer.
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer.

    """
    TODO: Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the 
    output layer according to the instructions. Define also the biases.
    """
    
    
    #initialises weights using a uniform distribution and rescales between layers
    
    W1=np.random.uniform(0,1,(n_hidden_layer,n_input_layer))
    W1=np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))

    W2=np.random.uniform(0,1,(n_output_layer,n_hidden_layer))
    W2=np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))
    

    # initialises biases with zeros
    
    bias_W1=np.zeros((n_hidden_layer,))
    bias_W2=np.zeros((n_output_layer,))


    # YOUR CODES ENDS HERE

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
 #   beta = 0.00005    #epsilon discount factor
    beta = 0.0001    #epsilon discount factor
 #   gamma = 0.85      #SARSA Learning discount factor
    gamma = 0.1      #SARSA Learning discount factor
    Alpha = 0.0001
    eta = 0.0035      #learning rate
    N_episodes = 50000 #Number of games, each game ends when we have a checkmate or a draw

    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction, 
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    
    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.    

    
    #variables to track the moves per game and reward per game
    R_save = np.zeros([N_episodes])
    N_moves_save = np.zeros([N_episodes])

    
    Average_Rewards = np.zeros([N_episodes])
    Average_moves = np.zeros([N_episodes])

    for n in range(N_episodes):
        epsilon_f = epsilon_0 / (1 + beta * n) #psilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

        #variable to store number of moves in a game
        Moves_Counter = 0

        while checkmate == 0 and draw == 0:
            R = 0  # Reward

            # Player 1

            # Actions & allowed_actions
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)

            # FILL THE CODE 
            # Enter inside the Q_values function and fill it with your code.
            # You need to compute the Q values as output of your neural
            # network. You can change the input of the function by adding other
            # data, but the input of the function is suggested. 
            Q, out1 = Q_values(x, W1, W2, bias_W1, bias_W2)

            """
            YOUR CODE STARTS HERE
            
            FILL THE CODE
            Implement epsilon greedy policy by using the vector a and a_allowed vector: be careful that the action must
            be chosen from the a_allowed vector. The index of this action must be remapped to the index of the vector a,
            containing all the possible actions. Create a vector calle da_agent that contains the index of the action 
            chosen. For instance, if a_allowed = [8, 16, 32] and you select the third action, a_agent=32 not 3.
            """
            
            
            #create array to contain Q values of possilbe actions
            Possible_Action = []
            
            #eps-greedy policy implementation
            Greedy = int(np.random.rand() > epsilon_f)             
            if Greedy:
                
               #put q values of possible actions into an array
               for i in allowed_a:
                    Possible_Action.append(Q[i])
            
               #get index of highest q value from possible actions
               Possible_Action = Possible_Action.index(max(Possible_Action))
               #use possible_index index value to select action
               action = allowed_a[Possible_Action]
            else:
                #Pick a random  allowed action
                action = np.random.choice(allowed_a)

            # selects action as that chosen by epsilon greedy
            a_agent = action  
            #THE CODE ENDS HERE. 


            # Player 1 makes the action
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]
                
                #increments move counter
                Moves_Counter += 1

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1  # Reward for checkmate

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, the agent gave checkmate.
                """
                
                # Backpropagation: output layer -> hidden layer
                out2delta = (R - Q[a_agent]) * np.heaviside(Q[a_agent], 0)
                #update weights and biases
                W2[a_agent] = (W2[a_agent] - (eta * out2delta * out1))
                bias_W2[a_agent] = (bias_W2[a_agent] - (eta * out2delta))
            
                # Backpropagation: hidden -> input layer
                out1delta = np.dot(W2[a_agent], out2delta) * np.heaviside(out1, 0)
                #update weights and biases
                W1 = W1 - (eta * np.outer(out1delta,x))
                bias_W1 = (bias_W1 -  (eta * out1delta))

                #set the reward for the game
                R_save[n] = R
                
                #calculate the running average of the reward per game
                Average_Rewards[n] = np.mean(R_save[:n])
                
                #increments move counter
                Moves_Counter += 1
                
                #set the number of moves for the game
                N_moves_save[n] = Moves_Counter
                #calculate the running average of the moves per game
                Average_moves[n] = np.mean(N_moves_save[:n])
                
                #calculate the exponential moving average of the reward
                if n > 0:
                    R_save[n] = ((1-Alpha) * R_save[n-1]) + (Alpha*R_save[n])
                    
                # THE CODE ENDS HERE

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, it is a draw.
                """  
                
                # Backpropagation: output layer -> hidden layer
                out2delta = (R - Q[a_agent]) * np.heaviside(Q[a_agent], 0)
                #update weights and biases
                W2[a_agent] = (W2[a_agent] - (eta * out2delta * out1))
                bias_W2[a_agent] = (bias_W2[a_agent] - (eta * out2delta))
            
                # Backpropagation: hidden -> input layer
                out1delta = np.dot(W2[a_agent], out2delta) * np.heaviside(out1, 0)
                #update weights and biases
                W1 = W1 - (eta * np.outer(out1delta,x))
                bias_W1 = (bias_W1 -  (eta * out1delta))

                #set the reward for the game
                R_save[n] = R
                
                #calculate the running average of the reward per game
                Average_Rewards[n] = np.mean(R_save[:n])
                
                #increments move counter
                Moves_Counter += 1
                
                #set the number of moves for the game
                N_moves_save[n] = Moves_Counter
                #calculate the running average of the moves per game
                Average_moves[n] = np.mean(N_moves_save[:n])
                
                #calculate the exponential moving average of the reward
                if n > 0:
                    R_save[n] = ((1-Alpha) * R_save[n-1]) + (Alpha*R_save[n])

                # YOUR CODE ENDS HERE
                if draw:
                    break

            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Update the parameters

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _ = Q_values(x_next, W1, W2, bias_W1, bias_W2)

            """
            FILL THE CODE
            Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
            rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
            the action made. You computed previously Q values in the Q_values function. Be careful: this is not the last 
            iteration of the episode, the match continues.
            """
            
            #increments move counter
            Moves_Counter += 1 
            
            # Backpropagation: output layer -> hidden layer
            out2delta = ((R + (gamma * np.max(Q_next)) - Q[a_agent]) * np.heaviside(Q[a_agent], 0))
            #update weights and biases
            W2[a_agent] = (W2[a_agent] - (eta * out2delta * out1))
            bias_W2[a_agent] = (bias_W2[a_agent] + (eta * out2delta))
        
            # Backpropagation: hidden -> input layer
            out1delta = np.dot(W2[a_agent], out2delta) * np.heaviside(out1, 0)
            #update weights and biases
            W1 = W1 - (eta * np.outer(out1delta,x))
            bias_W1 = (bias_W1 +  (eta * out1delta))

            # YOUR CODE ENDS HERE
            i += 1
    
    fontSize = 18

    print("Results for Q learning:")
    
    print("running average of the number of moves per game:")
    
    # plots the running average of the number of moves per game
    
    plt.plot(Average_moves)
    #set axis labels
    plt.xlabel('Number of episodes', fontsize = fontSize)
    plt.ylabel('Average Moves Per Game', fontsize = fontSize)
    #display plot
    plt.show()
    
    print("running average of the reward per game:")
    
    #plot running average of rewards
    #plt.plot(Average_Rewards)
    
    # plots the exponential moving average of the reward per game
    plt.plot(R_save)
    #set axis labels
    plt.xlabel('Number of episodes', fontsize = fontSize)
    plt.ylabel('Average Reward Per Game', fontsize = fontSize)
    #display plot
    plt.show()

if __name__ == '__main__':
    main()
