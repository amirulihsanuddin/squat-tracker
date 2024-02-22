Motion Tracker for Squat Exercise using Machine Learning

Introduction

Welcome to the Motion Tracker for Squat Exercise, a Final Year Project developed at UiTM. This project aims to evaluate the correctness of squat movements using machine learning models. The system assesses the squat's form and counts repetitions, providing users with real-time feedback.


How to Use
Run all for Working Prototype.ipynb and from there you have the options to choose a video or webcam

The Motion Tracker features two models:

Model 1: Focuses on squat repetitions and detects incorrect standing or squatting positions.

Model 2: Analyzes squat form, identifying issues such as bending, knee caving in, or caving out.


To switch webcams, modify the tracker() function in open_webcam within main.py.


All the correct dataset for the correct squat perfomed that was collected was approved and the incorrect squat movement was personally collected by myself


To visualize the squat performance graph, uncomment the accuracy array for the eight classes, the if-else statements, and the append statements for each class in the code. Run the code to generate graphs for each model.

#this is to remove a 0 when a 0 is detected between numbers

filtered_up_values = [up_accuracy[i] for i in range(len(up_accuracy)) if i == len(up_accuracy) - 1 or up_accuracy[i] != 0 or up_accuracy[i - 1] == 0]
filtered_down_values = [down_accuracy[i] for i in range(len(down_accuracy)) if i == len(down_accuracy) - 1 or down_accuracy[i] != 0 or down_accuracy[i - 1] == 0]
filtered_bend_values = [bend_accuracy[i] for i in range(len(bend_accuracy)) if i == len(bend_accuracy) - 1 or bend_accuracy[i] != 0 or bend_accuracy[i - 1] == 0]
filtered_cavein_values = [cavein_accuracy[i] for i in range(len(cavein_accuracy)) if i == len(cavein_accuracy) - 1 or cavein_accuracy[i] != 0 or cavein_accuracy[i - 1] == 0]
filtered_caveout_values = [caveout_accuracy[i] for i in range(len(caveout_accuracy)) if i == len(caveout_accuracy) - 1 or caveout_accuracy[i] != 0 or caveout_accuracy[i - 1] == 0]
filtered_good_values = [good_accuracy[i] for i in range(len(good_accuracy)) if i == len(good_accuracy) - 1 or good_accuracy[i] != 0 or good_accuracy[i - 1] == 0]
filtered_upbad_values = [upbad_accuracy[i] for i in range(len(upbad_accuracy)) if i == len(upbad_accuracy) - 1 or upbad_accuracy[i] != 0 or upbad_accuracy[i - 1] == 0]
filtered_downbad_values = [downbad_accuracy[i] for i in range(len(downbad_accuracy)) if i == len(downbad_accuracy) - 1 or downbad_accuracy[i] != 0 or downbad_accuracy[i - 1] == 0]

#to plot graph based on model 2

plt.plot(filtered_good_values, label='Good',fillstyle='none', linestyle='solid')
plt.plot(filtered_bend_values, label='Bend',fillstyle='none', linestyle='solid')
plt.plot(filtered_cavein_values, label='Cave In',fillstyle='none', linestyle='solid')
plt.plot(filtered_caveout_values, label='Cave Out ',fillstyle='none', linestyle='solid')

plt.xlabel('Frame')
plt.ylabel('Confidence')
plt.title('Confidence Tracking Throughout Video')
plt.legend(loc=('lower left'))
plt.show()

#to plot graph based on model 1

plt.plot(filtered_up_values, label='Up',fillstyle='none', linestyle='solid')
plt.plot(filtered_down_values, label='Down',fillstyle='none', linestyle='solid')
plt.plot(filtered_upbad_values, label='Up Bad',fillstyle='none', linestyle='solid')
plt.plot(filtered_downbad_values, label='Down Bad',fillstyle='none', linestyle='solid')


plt.xlabel('Frame')
plt.ylabel('Confidence')
plt.title('Confidence Tracking Throughout Video')
plt.legend(loc=('lower left'))
plt.show()
