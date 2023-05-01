# CS340 FINAL PROJECT
# Part B :Training an Artificial Neural network for bit string pattern classification !WITH EXTRA WORK
# Made by Nikola Jocic
# contact: 20200041@student.act.edu
# Description:
# Generates 1024 bit strings that are seperated into two files.
# One file that contains 80% is used for training AI, the other
# for testing the AI
# Captures all the training report, progress and generates a graph for the user


#Importing libraries
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import os

#Function for Option 1
def option1():

    #Generating the file bit_string that will be divided later
    with open('bit_strings.txt', 'w') as f:
        for i in range(1024):
            bit_strings = bin(i)[2:].zfill(10) #converting the number to bit and adding 0s
            f.write(bit_strings + '\n')

    #Open the file generated above and read the data
    with open('bit_strings.txt', 'r') as f:
        # Read all lines into a list
        all_lines = f.readlines()

        #Calculate amount of line for the training
        num_train = int(0.8 * len(all_lines))

        #Randomize the data in bit_string file to provide for a random selection
        random.shuffle(all_lines)

        #Seperate data in 20% and 80% and assign to appropriate lists
        train_data = all_lines[:num_train]
        input_data = all_lines[num_train:]

        #Write data into traiing file
        with open('training_data_unlabelled.txt', 'w') as f:
            f.writelines(train_data)

        #Write data into input file
        with open('input_data.txt', 'w') as f:
            f.writelines(input_data)

    #Reopen the file and store all the new lines as all_lines
    with open('training_data_unlabelled.txt', 'r') as f:
        all_lines = f.readlines()

    #Function to label lines
    def classify_bit_string(bit_string):
        num_ones_1 = bit_string[:5].count('1')
        num_ones_2 = bit_string[5:].count('1')
        if num_ones_1 > num_ones_2:
            return '10'
        elif num_ones_1 < num_ones_2:
            return '01'
        else:
            return '11'

    #Creating a new file and populating it using previous data and adding lables to it
    with open('training_data_labelled.txt', 'w') as f:
        for line in all_lines:
            line = line.strip()
            label = classify_bit_string(line)
            labeled_line = line + ',' + label + '\n' #Labeling
            f.write(labeled_line)

# Function for Option 2
def option2(input_nerons, hidden_neurons, output_neurons):

    #Setting up all the parameters from the user input
    input_size = input_nerons
    hidden_size = hidden_neurons
    output_size = output_neurons

    #Creating a machine learning model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='sigmoid', input_shape=(input_size,)),
        tf.keras.layers.Dense(output_size, activation='sigmoid')
    ])

    return model #Saving the model for further use

#Function for Option 3
def option3(model,filename, epochs,batch_number, l_rate):

    #Function that reads the data from the training file and seperates inputs and outputs into respective arrays
    def read_training_data(file_path):
        inputs = []
        outputs = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                data = line.split(',')
                inputs.append([int(x) for x in data[0]])
                outputs.append([int(x) for x in data[1]])
        return np.array(inputs), np.array(outputs)  #Returning the arrays

    #Running the function and storing inputs and outputs
    inputs, outputs = read_training_data(filename)

    #Compiling the model
    opt = keras.optimizers.Adam(learning_rate=l_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy',  tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    #Training the model
    batch_size = batch_number   #EXTRA WORK DEFINING THE BATCH SIZE FOR MORE EFFICIENT LEARNING
    num_epochs = epochs
    history = model.fit(inputs, outputs, batch_size=batch_size, epochs=num_epochs,verbose=0)

    #Saving weights in memory
    weights= tf.keras.Model.save_weights

    #Creating new file "training_progress.txt" in order to store the training progress
    new_file = 'training_progress.txt'
    with open(new_file, 'w') as f:
        #Storing the epoch and loss function
        for i, loss in enumerate(history.history['loss']):
            if i % 10 == 0:
                f.write(f"{i},{loss}\n")

    #Confirming the saving of the files
    print(f"The epoch number and loss function value were saved to {new_file}")

    #Making the report of training
    predictions = model.predict(inputs)
    rounded_predictions = np.round(predictions)
    accuracy = history.history['accuracy'][-1]
    precision= history.history['precision'][-1]
    recall = history.history['recall'][-1]
    confusion_matrix = tf.math.confusion_matrix(outputs.argmax(axis=1), rounded_predictions.argmax(axis=1)).numpy()
    report = f'Training Report:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nConfusion Matrix:\n{confusion_matrix}'
    print(report) #Printing the report
    model.save('trainedmodel') #Saving the model for further use

#Function for Option 4
def option4():

    #Importing the saved model from the file and storing it
    model = keras.models.load_model('trainedmodel')

    #Function to read the input data bits and convert them to ints
    def read_training_data(file_path):
        test_inputs = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                test_inputs.append([int(x) for x in line])
        return np.array(test_inputs)

    #Storing the infomration from input data as an array into test_inputs
    test_inputs = read_training_data("input_data.txt")

    #Using the test_inputs and the model to predict the output
    test_output = model.predict(test_inputs)

    #Rounding the output
    test_output = np.round(test_output)

    #Saving the outputs next to inputs in a new file
    new_file = 'training_output.txt'
    with open(new_file, 'w') as f:
        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            predicted_output = test_output[i]
            f.write(str(test_input)+","+str(predicted_output)+"\n")

    #Confirming the file is saved
    print(f"The test inputs and predicted outputs were saved to {new_file}")

#Function for Option 5
def option5():

    #Defining the axis
    x_axis = []
    y_axis = []

    #Opening the training_progress file in order to graph the training progress
    with open("training_progress.txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            x_axis.append(int(currentline[0]))
            y_axis.append(float(currentline[1]))

    #Ploting the graph and adding details
    plt.plot(x_axis, y_axis)
    plt.xlabel('TRAINING EPOCHS')
    plt.ylabel('COST FUNCTION OUTPUT')
    plt.title('TRAINING PROGRESS')
    plt.tight_layout()
    plt.show()

#Function for the menu
def menu():
    while 1:
        try:  # checks if the input is an integer
            choice = int(input('''\n
    1.) Generate training and testing data files
    2.) Choose a network topology
    3.) Initiate a series of training passes, generate training report
    4.) Classify test data
    5.) Display training progression graph
    6.) Exit the program \n
    your choice: ''').strip())
            # presents the options of the menu and strips the input of the white spaces in front
            # and after

        except ValueError:  # in case the input is not an integer thn display the appropriate message
            print(" Please input an integer")

        else:  # in case no error is raised then proceed with the rest of the code
            if choice == 1:  # checks if the user selected option 1
                print('Working on option 1...')
                option1()  # calls option 1

            elif choice == 2:  # checks if the user selected option 2
                print('Working on option 2...')
                hidden_neurons = int(input("Please provide hidden layer amount: "))
                #declaring parameters
                input_neurons = 10
                output_neurons = 2
                option2(input_neurons, hidden_neurons, output_neurons)  # calls option 2
                print("The used topology will be 10-" + str(hidden_neurons) + "-2")

            elif choice == 3:  # checks if the user selected option 3
                print('Working on option 3...')
                #default file path
                default_file = 'training_data_labelled.txt'
                #Setting up a default input in case user does not input anything
                file_name = input(f"Enter the training data set file name (default: {default_file}): ")
                if file_name == '':
                    file_name = default_file
                #Ask the user for the file name
                while not os.path.exists(file_name):
                    file_name = input(f"Enter the training data set file name (default: {default_file}): ")
                    if file_name == '':
                        file_name = default_file
                #default learning step
                default_step = 0.001
                #Ask the user for a rate of learning
                # Setting up a default input in case user does not input anything
                step = input(f"Enter the learning step (default: {default_step}): ")
                if step == '':
                    step = default_step
                else:
                    step = float(step)
                #Setting up default epochs
                # Setting up a default input in case user does not input anything
                default_epochs = 150
                epochs = input(f"Enter the amount of epochs (default: {default_epochs}): ")
                if epochs == '':
                    epochs = default_epochs
                else:
                    epochs = int(epochs)
                #Setting up default batch
                # Setting up a default input in case user does not input anything
                default_batch = 32
                batch = input(f"Enter the batch size (default: {default_batch}): ")
                if batch == '':
                    batch = default_batch
                else:
                    batch = float(batch)
                # calls option 3
                option3(option2(input_neurons, hidden_neurons, output_neurons),file_name, epochs, batch, step)

            elif choice == 4:  # checks if the user selected option 4
                print('Working on option 4...')
                option4()  # calls option 4

            elif choice == 5:  # checks if the user selected option 5
                print('Working on option 5...')
                option5()  # calls option 5

            elif choice == 6:  # checks if the user selected option 6
                print('Hope that this was a joyful experience!\n\n')
                break  # breaks from the loop effectively ending the program

            else:  # in case the input is an integer, but it doesn't represent any of the options
                print('    Please input an integer corresponding to one of the options \n')


print('''\n\n\n ANN TRAINING FOR BIT STRING CLASSIFICATION''')
menu()  # calls the menu

