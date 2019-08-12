import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *
np.random.seed(77)
size_board = 4

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
    n_input_layer = 50  # Number of neurons of the input layer. TODO: Change this value
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer. TODO: Change this value accordingly

    """
    TODO: Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the 
    output layer according to the instructions. Define also the biases.
    """
    # Weights matrix, connecting input neurons (state) to hidden layers (actions). Initially random
    
    # W1 is defined and resclaed by the totla number of connections between the consdiered two layers
    W1=np.random.uniform(0,1,(n_hidden_layer,n_input_layer))    
    W1=np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))
   
    # W1 is defined and resclaed by the totla number of connections between the consdiered two layers
    W2=np.random.uniform(0,1,(n_output_layer,n_hidden_layer))
    W2=np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))

    # bias is set to zero
    bias_W1=np.zeros((n_hidden_layer,))
    bias_W2=np.zeros((n_output_layer,))



    # YOUR CODES ENDS HERE

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005 # 0.005     #epsilon discount factor
    gamma = 0.85  #0.15      #SARSA Learning discount factor
    eta = 0.0035  #0.0035      #learning rate
    N_episodes = 1000 #Number of games, each game ends when we have a checkmate or a draw
    rmsprop = False
    ###  Training Loop  ### 
    #varialbe setting for RMSprop calculation
    W2_average = 0
    W1_average = 0
    W2_bias_average = 0
    W1_bias_average=0
    eta_w2 = 0
    eta_w1 = 0
    eta_bias1 = 0
    eta_bias2 = 0

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

   # R_save = np.zeros([N_episodes, 1])
    #N_moves_save = np.zeros([N_episodes, 1])
    R_save = np.zeros([N_episodes])
    N_moves_save = np.zeros([N_episodes])
    alpha = 0.0001 # for exponential moving average
    # END OF SUGGESTIONS

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
       
        moves_game = 0 # to store moves per game

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
            containing all the possible actions. Create a vector called a_agent that contains the index of the action 
            chosen. For instance, if a_allowed = [8, 16, 32] and you select the third action, a_agent=32 not 3.
            """           

            #eps-greedy policy implementation
            greedy = (np.random.rand() > epsilon_f)           
            if greedy:
                #a_agent = allowed_a[np.take(Q, allowed_a).tolist().index(max(np.take(Q, allowed_a).tolist()))]
                a_agent = allowed_a[np.argmax(np.take(Q, allowed_a))]  #pick the best action
            else:
                a_agent = np.random.choice(allowed_a)                  #pick random action
            
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
                # One more move is made from player 1              
                moves_game += 1
            else:
                direction = a_agent - possible_queen_a
                steps = 1
                
                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]
                # One more move is made from player 1
                moves_game += 1
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

                #Backpropagation
                # calcuating delta and update weight and bias for W2 
                # if statements indicates the heaviside function
                out1delta = np.dtype(np.complex128)
                #Backpropagation for the output layer
                if ((np.dot(W2[a_agent], out1)) > 0):
                    out2delta = (R - Q[a_agent])
                    W2[a_agent] += (eta * out2delta * out1) 
                    bias_W2 += (eta * out2delta)
                    #calculating backpropagationg for hidden layer
                    if (np.sum(np.dot(W1, x)) > 0):
                        out1delta = np.dot(W2[a_agent], out2delta)
                        W1 += (eta * np.outer(out1delta, x))
                        bias_W1 += (eta * out1delta)

                # It is checkmate, plot rewards and moves per game (exponential moving average), alpha = 0.0001
                if n >0:
                    R_save[n] = ((1-alpha) * R_save[n-1]) + (alpha*R)
                else:
                    R_save[n] = R

                if n >0:
                    N_moves_save[n] = ((1-alpha) * N_moves_save[n-1]) + (alpha*moves_game)
                else:
                    N_moves_save[n] = moves_game
                
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
                
                out1delta = np.dtype(np.complex128)
                #Backpropagation, same as above
                if ((np.dot(W2[a_agent], out1)) > 0):
                    out2delta = (R - Q[a_agent])
                    W2[a_agent] += (eta * out2delta * out1) 
                    bias_W2 += (eta * out2delta)
                    if (np.sum(np.dot(W1, x)) > 0):
                        out1delta = np.dot(W2[a_agent], out2delta)
                        W1 += (eta * np.outer(out1delta, x))
                        bias_W1 += (eta * out1delta)

                
                # It is draw, plot rewards and moves per game (exponential moving average), alpha = 0.0001
                if n >0:
                    R_save[n] = ((1-alpha) * R_save[n-1]) + (alpha*R)
                else:
                    R_save[n] = R

                if n >0:
                    N_moves_save[n] = ((1-alpha) * N_moves_save[n-1]) + (alpha*moves_game)
                else:
                    N_moves_save[n] = moves_game
                
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
                # one more move
                moves_game += 1

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
            # error cost function
            # error = 0.5*((R- (gamma*np.max(Q_next))-Q[a_agent])**2)
            # print(error)

            # FOR SARSA - Reapply epsilong greedy policy for Q_next
            # a_new = np.concatenate([np.array(a_q1), np.array(a_k1)])
            # allowed_a = np.where(a_new > 0)[0]

            ##eps-greedy policy implementation
            # greedy = (np.random.rand() > epsilon_f)           
            # if greedy:
            #     #a_agent = allowed_a[np.take(Q, allowed_a).tolist().index(max(np.take(Q, allowed_a).tolist()))]
            #     next_agent = allowed_a[np.argmax(np.take(Q, allowed_a))]  #pick the best action
            # else:
            #     next_agent = np.random.choice(allowed_a)                  #pick random action

        

            # out1delta = np.dtype(np.complex128) # to preven to overflowing issues
            # #Backpropagation, same as above but this time the next Q-value is considered.
            # #RMSprop is activated when it is set to True
            # if ((np.dot(W2[a_agent], out1)) > 0):
            #     out2delta = (R - Q[a_agent] + gamma * np.max(Q_next))
            #     #FOR SARSA
            #     # out2delta = (R - Q[a_agent] + gamma * Q_next(next_agent))
            #     out1delta = np.dot(W2[a_agent], out2delta)
            #     W1_d = np.outer(x, out1delta)
            #     W2_d = np.outer(out1, out2delta)
            #     if rmsprop:
            #         alpha_rms = 0.9 #  a recmommend value
            #         # The calculation of RMSProp
            #         W2_average = (alpha_rms * W2_average) +(1.0 - alpha_rms) * (W2_d)**2
            #         W1_average = (alpha_rms * W1_average) +(1.0 - alpha_rms) * (W1_d)**2
            #         W2_bias_average = (alpha_rms * W2_bias_average) +(1.0 - alpha_rms) * (out2delta)**2
            #         W1_bias_average = (alpha_rms * W1_bias_average) +(1.0 - alpha_rms) * (out1delta)**2
            #         # applying different learning rates
            #         eta_w2 = eta/ np.sqrt(W2_average[a_agent])
            #         eta_w1 = eta / np.sqrt(W1_average)
            #         eta_bias2 = eta/ W2_bias_average
            #         eta_bias1 = eta/ W1_bias_average
            #     W2[a_agent] += (eta_w2 * out2delta * out1) 
            #     bias_W2 += (eta_bias2 * out2delta)
            #     #backpropagation for the hidden layer
            #     if (np.sum(np.dot(W1, x)) > 0):
            #         # W1 += np.outer(x, out1delta).T * eta_w1.T
            #         # bias_W1 += (eta_bias1 * out1delta)
            # else: # without rmsprop, just normal backpropagation as before is applied
            out1delta = np.dtype(np.complex128)
            if ((np.dot(W2[a_agent], out1)) > 0):
                out2delta = (R - Q[a_agent]+ gamma * np.max(Q_next))
                W2[a_agent] += (eta * out2delta * out1) 
                bias_W2 += (eta * out2delta)
                if (np.sum(np.dot(W1, x)) > 0):
                    out1delta = np.dot(W2[a_agent], out2delta)
                    W1 += (eta * np.outer(out1delta, x))
                    bias_W1 += (eta * out1delta)                


            # match continues, so one more move
            moves_game += 1

            # YOUR CODE ENDS HERE
            i += 1
    
    fontSize = 12

    plt.plot(R_save)
    plt.xlabel('Nth Game', fontsize=fontSize)
    plt.ylabel('Rewards per Game (Defalut)', fontsize=fontSize)
    plt.show()
    
    plt.plot(N_moves_save)
    plt.xlabel('Nth Game', fontsize=fontSize)
    plt.ylabel('moves per Game (Defalut)', fontsize=fontSize)
    plt.show()

if __name__ == '__main__':

    main()
