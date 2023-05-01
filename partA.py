# CS340 FINAL PROJECT
# Part A: Qatar 2022 World Cup top scorers: football data processing !WITH EXTRA WORK
# Made by Nikola Jocic
# contact: 20200041@student.act.edu
# Description:
# Set of 7 menu options where the user will be provided with visualization
# of the statistics from the latest football world cup (2022).

#Importing libraries
import csv
from beautifultable import BeautifulTable
import matplotlib.pyplot as plt
import os

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

#Opening the file and reading in and saving data
with open('top_scorers_Qatar2023.csv', 'r') as file:
    dataList = list(csv.reader(file, delimiter=","))
    file.close()

    #Making global tables to be used later
    table2 = BeautifulTable()
    table1 = BeautifulTable()
    table3 = BeautifulTable()

    for i in dataList:
        table1.rows.append(i)

def option1(): #Funciton for Option 1 displays the read data
    print(table1)

def option2(): #Funciton for Option 2

    #Sorts the table accorind to the appropriate column
    table1.rows.sort(7, reverse=True)

    #Requests the user to input a threshold
    threshold = float(input("Please input a threshold-average for the shot distance: "))

    #Sorts the table according to the threshold parameter
    tableTemp = BeautifulTable()
    for x in dataList:
        if x[7] == "AVG_SHOT_DISTANCE":
            continue
        elif float(x[7]) > threshold:
            tableTemp.rows.append(x)

    #Displays only from the threshold and above in descending order
    tableTemp.rows.sort(0)
    print(tableTemp)

def option3(): #Function for Option 3

    #Creating a new temporary list in order to store newly calculated data
    calculation_data = []

    #Setting the columns of the new table
    table2.columns.header = ['PLAYER', 'TEAM', 'AGE', 'GOALS', 'GOALS FROM PENALTY', 'SHOTS',
                             'SHOTS_ON_TARGET', 'AVG_SHOT_DISTANCE', 'SHOTS_GOALS',
                             'NON-PENALTY-SHOTS-GOALS']
    # taking out the 'headers'(first element of the list) from the dataList variable which is used to populate the table
    dataList.pop(0)

    #Populating the temporary list created earlier  and making neccessary calculations
    for i in dataList:
        calculation_data.append((i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], (round(int(i[3]) / (int(i[5])), 1)),
                                 (round((int(i[3]) - int(i[4])) / int(i[5]), 1))
                                 ))
    #Populating the table with the newly calculated content
    for p in calculation_data:
        table2.rows.append(p)

    #Saving the table in a txt file while accomodating for the headers to preserve table format
    with open('jocic_qatar2022_stats.txt', mode='w') as file:
        file.write("\t".join(table2.columns.header) + "\n")

        for row in table2:
            rows = [str(cell) for cell in row]
            file.write("\t".join(rows) + "\n")

    #Reopening the table
    with open('jocic_qatar2022_stats.txt', mode='r') as file:
        lines = file.readlines()

    #Accomodating for the headers and re-populating the table with the content from the file
    headers = lines[0].strip().split('\t')
    table3.columns.header = headers
    for line in lines[1:]:
        fields = line.strip().split('\t')
        table3.rows.append(fields)

    # Print the table
    print(table3)

    #For further use, returns to global
    return table3


def option4(): # Function for Option 4, !RUN THE OPTION  3 FIRST!

    #Making needed variables for the graph (x and y coordinates)
    players = list(table3.columns['PLAYER'])
    shots = list([float(field) for field in table3.columns['SHOTS_GOALS']])
    goals = list([float(field1) for field1 in table3.columns['NON-PENALTY-SHOTS-GOALS']])


    #Making a figure that will contain two graphs (pl1, pl2)
    figure, (pl1, pl2)=plt.subplots(2, 1, figsize=(10, 5))

    #Configuring the design of both graphs
    pl1.bar(players, shots, width=0.1, color=['orange', 'blue'])
    pl1.tick_params(axis='x', labelsize=6)
    pl2.bar(players, goals, width=0.3, color=['brown', 'purple'])
    pl2.tick_params(axis='x', labelsize=6)

    #Configuring the layout of the figure as well as plotting the two graphs
    plt.tight_layout()
    plt.show()


def option5(column, value): #Function for Option 5 !RUN THE OPTION  3 FIRST!
    def orderCheck(value): #Checks the whether to order in ascending or descending and saves the bool
        if value == 1:
            return True
        elif value == 0:
            return False

    #creating a new table in order to preserve changes within this option
    table4 = BeautifulTable()
    with open('jocic_qatar2022_stats.txt', mode='r') as file:
        lines = file.readlines()

    #accomodating for headers and populating new table
    headers = lines[0].strip().split('\t')
    table4.columns.header = headers
    for line in lines[1:]:
        fields = line.strip().split('\t')
        table4.rows.append(fields)

    #Prints the table sorted according to user input
    table4.sort(column - 1, reverse= orderCheck(value))
    print(table4)

#EXTRA WORK!!!!
def option6(): #Function for Option 6

    #New table in order to seperate extra work
    table5=BeautifulTable()
    table6 = BeautifulTable()
    calculation_data2 = []

    #Defining headers of the old table
    table6.columns.header = ['PLAYER', 'TEAM', 'AGE', 'GOALS', 'GOALS FROM PENALTY', 'SHOTS',
                             'SHOTS_ON_TARGET', 'AVG_SHOT_DISTANCE', 'ACCURACY %']

    # taking out the 'headers'(first element of the list) from the dataList variable which is used to populate the table
    dataList.pop(0)

    #Populating new temporary list and making neccessary calculations
    for i in dataList:
        calculation_data2.append((i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], round((float(i[3])/float(i[5])*100), 1) ))

    #populating old table
    for p in calculation_data2:
        table6.rows.append(p)

    #Creating a new file that will display the extra work
    with open('newStat.txt', mode='w') as file:
        file.write("\t".join(table2.columns.header) + "\n")

        for row in table2:
            rows = [str(cell) for cell in row]
            file.write("\t".join(rows) + "\n")

    #Reopening the same file and populating the new table in order to present new columns
    with open('newStat.txt', mode='r') as file:
        lines = file.readlines()

    headers = lines[0].strip().split('\t')
    table5.columns.header = headers
    for line in lines[1:]:
        fields = line.strip().split('\t')
        table5.rows.append(fields)

    # Print the table
    print(table5)

#MENU CODE
def menu():
    while 1:
        try:  # checks if the input is an integer
            choice = int(input('''\n
        1.) Read & display stats for the top goal scorers
        2.) Display player shooting distance avg past a threshold, in alphabetical order
        3.) Calculate G/S and NPG/S metadata, save into a new file, display 
        4.) Visualize G/S and NPG/S metadata
        5.) Sort by a field indicated by the user
        6.) Calculate Accuracy of shots taken by each player, display and save into a new file 
        7.) Exit. \n
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
                print(' Working on option 2...')
                option2()  # calls option 2

            elif choice == 3:  # checks if the user selected option 3
                print(' Working on option 3...\n\n')
                option3()

            elif choice == 4:  # checks if the user selected option 4
                print(' Working on option 4...\n\n')
                file_path = "jocic_qatar2022_stats.txt"

                # Check if the file exists
                if os.path.isfile(file_path):
                    option4()
                else:
                    print("Please run option 3 beforehand")


            elif choice == 5:  # checks if the user selected option 5
                print(' Working on option 5...\n\n')
                file_path = "jocic_qatar2022_stats.txt"
                if os.path.isfile(file_path):
                    columnChoice = int(input("Please tell us according to  which column you want the table to be sorted "
                                             "(1.PLAYER, 2.TEAM, 3.AGE, 4.GOALS, 5.GOALS FROM PENALTY,"
                                             " 6.SHOTS, 7.SHOTS_ON_TARGET, "
                                             "8.AVG_SHOT_DISTANCE, 9.SHOTS_GOALS, 10.NON-PENALTY-SHOTS-GOALS) :"))
                    orderChoice = int(input("Please pick order, type either 1 or 0 for descending or ascending:"))
                    option5(columnChoice, orderChoice)
                else:
                    print("Please run option 3 beforehand")

            elif choice == 6:  # checks if the user selected option 6
                print(' Working on option 6...')
                option6()  # calls option 6

            elif choice == 7:  # checks if the user selected option 7
                print('Hope that you found the information you were looking for!\n\n')
                break  # breaks from the loop effectively ending the program

            else:  # in case the input is an integer, but it doesn't represent any of the options
                print('    Please input an integer corresponding to one of the options \n')

print('''\n\n\n TOP SCORER STATISTICS FROM THE FOOTBALL WORLD CUP 2022''')
menu()  # calls the menu