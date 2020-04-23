from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



"""** The data set is in the form of array with key, so to make machine learning the data should be in the form of vector,
    packing the data set in vector format **
    """

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def pack_features_vector_test(features):
    features = tf.stack(list(features.values()), axis=1)
    return features


"""** defining the loss function and optimizer to reduce the prediction error or loss"""

def loss(model, x, y):
  y_ = model(x)
  return tf.compat.v1.losses.sparse_softmax_cross_entropy(y, y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Tkinter main
root = Tk()
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.width',None)
#setting the title, size and color of the window
root.title("Anomaly detection in smart grid")
root.geometry('950x480')
root.configure(background='#2b2b2b')

#setting tab layout
tab = ttk.Notebook(root)


image_t1 = ImageTk.PhotoImage(Image.open("D:\Photoshop\Phoshop edits\project bg1png.png"))
image_t2 = ImageTk.PhotoImage(Image.open("D:\Photoshop\Phoshop edits\project bg2png.png"))

tab_1 = Canvas(tab)
tab_1.create_image(0, 0, anchor=NW, image=image_t1)
tab.add(tab_1, text="Train Module")

tab_2 = Canvas(tab, background='#313335')
tab_2.create_image(0, 0, anchor=NW, image=image_t2)
tab.add(tab_2, text="Test Module")

tab.pack(expand=2, fill='both')


# ADDING BUTTONS and label TO TRAIN MODULE
t_label = Label(tab_1, text="Train Dataset", bg="#2b2b2b", fg='white', width=9, height=2)
t_label.grid(row=3, column=2, padx=20, pady=30)

# dialog list box
e_ent = Text(tab_1, width=35, height=1)
e_ent.grid(row=3, column=3, columnspan=3)

# setting dialogbox for opening file function
def open_file():
    global traindata_loc
    root.filename = filedialog.askopenfilename(initialdir="c:/", title="Select the dataset", filetype=(("CSV file", "*.csv"),))
    traindata_loc = root.filename
    output_box.config(state="normal")
    output_box.delete('1.0', END)
    e_ent.insert(INSERT, traindata_loc)
    e_ent.config(state="disabled")
    output_box.insert(INSERT, "DATASET SUCCESSFULLY LOADED")
    output_box.config(state="disabled")

dialog_b = Button(tab_1, text=". . .",  bg="#3c3f41", fg='white', width=12, height=1, command=open_file)
dialog_b.grid(row=3, column=6, padx=50,pady=30)

# Show dataset button function
def show_dataset():
    global train_dataset_fp
    train_dataset_fp = pd.read_csv(traindata_loc)
    output_box.config(state="normal")
    output_box.delete('1.0', END)
    output_box.insert(INSERT,train_dataset_fp)
    output_box.config(state="disabled")
    print(train_dataset_fp)

# Cleaning the dataset function
def clean_dataset():
    global  IP_col, dic, ID_col, ID_dic, refined_data, train_data, val, val_ID, feature_names, label_name, batch_size, train_column_names
    # float convertion of IP
    IP_col = list(train_dataset_fp['Requested_IP'])
    IP_col = list(dict.fromkeys(IP_col))
    dic = {}
    val = 0.0
    for l in range(1, len(IP_col) + 1):
        val = val + 0.1
        st = IP_col[l - 1]
        d = {st: round(val, 1)}
        dic.update(d)
    train_dataset_fp.Requested_IP = [dic[(item)] for item in train_dataset_fp.Requested_IP]

    # float convertion of ID
    ID_col = list(train_dataset_fp['ID_Details'])
    ID_col = list(dict.fromkeys(ID_col))
    ID_dic = {}
    val_ID = 0.0
    for l in range(1, len(ID_col) + 1):
        val_ID = val_ID + 0.1
        new_st = ID_col[l - 1]
        new_d = {new_st: round(val_ID, 1)}
        ID_dic.update(new_d)
    train_dataset_fp.ID_Details = [ID_dic[(item)] for item in train_dataset_fp.ID_Details]

    # float convertion of day
    train_dataset_fp.Day = [float(i) for i in train_dataset_fp.Day]

    # execute after refining the dataset
    n_col = ['ID_Details', 'Requested_IP', 'Consumption', 'Day', 'label']
    refined_data = train_dataset_fp[n_col]
    refined_data.to_csv("refined_train_dataset.csv", index=False)
    train_data = 'refined_train_dataset.csv'

    global features, labels, model, optimizer, train_loss_results, train_accuracy_results, global_step, train_dataset

    # column order in CSV file
    train_column_names = ['ID_Details', 'Requested_IP', 'Consumption', 'Day', 'label']

    feature_names = train_column_names[:-1]
    label_name = train_column_names[-1]

    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))

    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(train_data, batch_size, column_names=train_column_names,
                                                          label_name=label_name, num_epochs=1, shuffle=True)

    features, labels = next(iter(train_dataset))

    """** The data set is in the form of array with key, so to make machine learning the data should be in the form of vector,
    packing the data set in vector format **
    """

    train_dataset = train_dataset.map(pack_features_vector)

    features, labels = next(iter(train_dataset))

    print(features[:5], labels[:5])

    """** Making the machine learning model with activation function **"""

    model = tf.keras.Sequential([tf.keras.layers.Dense(15, activation=tf.nn.relu, input_shape=(4,)),
                                 tf.keras.layers.Dense(12, activation=tf.nn.relu), tf.keras.layers.Dense(2)
                                 ])

    predictions = model(features)
    predictions[:5]

    tf.nn.softmax(predictions[:5])

    print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
    print("    Labels: {}".format(labels))
    l = loss(model, features, labels)
    print("Loss test: {}".format(l))

    optimizer = tf.optimizers.SGD(learning_rate=0.01)

    global_step = tf.Variable(0)

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(global_step.numpy(), loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    print("Step: {},         Loss: {}".format(global_step.numpy(), loss(model, features, labels).numpy()))

    train_loss_results = []
    train_accuracy_results = []

    # Disabling the clean button
    clean_b.config(state="disabled")

    output_box.config(state="normal")
    output_box.delete('1.0', END)
    output_box.insert(INSERT, "DATASET SUCCESSFULLY REFINED")
    output_box.config(state="disabled")
    print("DATASET SUCCESSFULLY REFINED")
    print(val)

# Displaying refined dataset function
def show_refined():
    output_box.config(state="normal")
    output_box.delete('1.0', END)
    output_box.insert(INSERT, refined_data)
    output_box.config(state="disabled")
    print(refined_data)

# Train machine learning model function
def train_ml():
    output_box.config(state="normal")
    output_box.delete('1.0', END)
    accuracy_lis = []
    num_epochs = 226

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.metrics.Mean()
        epoch_accuracy = tf.metrics.Accuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
            acc = "Error Rate: {:.3f}  |  Accuracy: {:.3%}".format(epoch_loss_avg.result(),
                                                                        epoch_accuracy.result())
            accuracy_lis.append(acc)
    for l in accuracy_lis:
        output_box.insert(INSERT, l + '\n')
    output_box.config(state="disabled")

def visual_graph():
    #figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig, axes = plt.subplots(2, sharex=True, figsize=(8, 4))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()


show_d = Button(tab_1, text="Show Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=show_dataset)
show_d.grid(row=3, column=9, padx=100, pady=20)

# cleanup button
clean_b = Button(tab_1, text="Clean Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=clean_dataset)
clean_b.grid(row=5, column=9, padx=100, pady=15)

# Show refined dataset button
refined_b = Button(tab_1, text="Refined Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=show_refined)
refined_b.grid(row=7, column=9, padx=100, pady=15)

# train button
train_b = Button(tab_1, text="Train Machine", bg="#3c3f41", fg='white', width=12, height=2, command=train_ml)
train_b.grid(row=9, column=9, padx=100, pady=15)

# visual button
visual_b = Button(tab_1, text="Visualize", bg="#3c3f41", fg='white', width=12, height=2, command=visual_graph)
visual_b.grid(row=11, column=9, padx=100, pady=15)

# output box
global output_box
output_box = Text(tab_1, height=20, width=70,  background='#313335', foreground='white')
output_box.insert(INSERT, "Output Box")
output_box.config(state="disabled",wrap='none')
output_box.grid(row=5, column=3, columnspan=5, rowspan=8,pady=0)

# adding scroll bar on output window
scrolly = Scrollbar(tab_1,  background='#313335')
scrolly.grid(row=12, column=8, rowspan=2)

scrollx = Scrollbar(tab_1, orient='horizontal', background='#313335')
scrollx.grid(row=13, column=7, columnspan=2, padx=17)

scrolly.config(command=output_box.yview)
scrollx.config(command=output_box.xview)

output_box.config(xscrollcommand=scrollx.set, yscrollcommand=scrolly.set)


'''------------------------------------------------ TRAIN MODULE ----------------------------------------------------'''
# ADDING BUTTONS and label TO TEST MODULE
test_label = Label(tab_2, text="Test Dataset", bg="#2b2b2b", fg='white', width=9, height=2)
test_label.grid(row=3, column=2, padx=20, pady=30)

# dialog list box
test_e_ent = Text(tab_2, width=30, height=1)
test_e_ent.grid(row=3, column=3, columnspan=3)


# setting dialogbox for opening file function
def test_open_file():
    global testdata_loc
    root.filename = filedialog.askopenfilename(initialdir="c:/", title="Select the dataset", filetype=(("CSV file", "*.csv"),))
    testdata_loc = root.filename
    test_output_box.config(state="normal")
    test_output_box.delete('1.0', END)
    test_e_ent.insert(INSERT, testdata_loc)
    test_e_ent.config(state="disabled")
    test_output_box.insert(INSERT, "TESTING DATASET SUCCESSFULLY LOADED")
    test_output_box.config(state="disabled")


test_dialog_b = Button(tab_2, text=". . .",  bg="#3c3f41", fg='white', width=12, height=1, command=test_open_file)
test_dialog_b.grid(row=3, column=6, padx=50,pady=30)


# Show dataset button function
def test_show_dataset():
    global test_dataset_fp, Final_test_data
    test_dataset_fp = pd.read_csv(testdata_loc)
    Final_test_data = pd.read_csv(testdata_loc)
    test_output_box.config(state="normal")
    test_output_box.delete('1.0', END)
    test_output_box.insert(INSERT,test_dataset_fp)
    test_output_box.config(state="disabled")
    print(test_dataset_fp)

# Refining the test data
def test_clean_dataset():
    global val, val_ID, features, labels
    """*** Perforing data pre-processing by converting the IP ***"""
    global Test_IP_col, Test_ID_col, refined_test_data, test_dataset

    Test_IP_col = list(test_dataset_fp['Requested_IP'])
    Test_IP_col = list(dict.fromkeys(Test_IP_col))
    for ip in Test_IP_col:
        if ip not in IP_col:
            # Assigning new value to IP if the IP is new
            val = val + 0.1
            st = ip
            d = {st: round(val, 1)}
            dic.update(d)


    test_dataset_fp.Requested_IP = [dic[(item)] for item in test_dataset_fp.Requested_IP]


    """*** Converting the id ***"""

    # float convertion of ID
    Test_ID_col = list(test_dataset_fp['ID_Details'])
    Test_ID_col = list(dict.fromkeys(Test_ID_col))
    for id in Test_ID_col:
        if id not in ID_col:
            # Assigning new value to ID if the ID is new
            val_ID = val_ID + 0.1
            new_st = id
            new_d = {new_st: round(val_ID, 1)}
            ID_dic.update(new_d)

    test_dataset_fp.ID_Details = [ID_dic[(item)] for item in test_dataset_fp.ID_Details]


    """*** Converting day ***"""

    test_dataset_fp.Day = [float(i) for i in test_dataset_fp.Day]


    """*** Forming the refined Dataset ***"""

    n_col = ['ID_Details', 'Requested_IP', 'Consumption', 'Day']
    refined_test_data = test_dataset_fp[n_col]
    refined_test_data.to_csv("refined_test_dataset.csv", index=False)

    test_data = 'refined_test_dataset.csv'

    test_column_names = ['ID_Details', 'Requested_IP', 'Consumption', 'Day']

    """*** packing the dataset ***"""

    test_dataset = tf.data.experimental.make_csv_dataset(test_data, batch_size, column_names=test_column_names,
                                                         num_epochs=1, shuffle=False)

    test_chart = test_dataset

    features = next(iter(test_dataset))
    global plotcategory
    plotcategory = features
    # category = labels

    print(plotcategory)

    test_dataset = test_dataset.map(pack_features_vector_test)


    test_clean_b.config(state="disabled")

    test_output_box.config(state="normal")
    test_output_box.delete('1.0', END)
    test_output_box.insert(INSERT, "TESTING DATASET SUCCESSFULLY REFINED")
    test_output_box.config(state="disabled")
    print("TESTING DATASET SUCCESSFULLY REFINED")

# showing the refined test dataset
def test_show_refined():
    test_output_box.config(state="normal")
    test_output_box.delete('1.0', END)
    test_output_box.insert(INSERT, refined_test_data)
    test_output_box.config(state="disabled")
    print(refined_data)

# detect anomaly in testing dataset
def detect_test():
    """*** Detecting the anomaly by passing the dataset to the Machine learning model ***"""
    global prediction
    test_accuracy = tf.metrics.Accuracy()

    for x in test_dataset:
        print(x)

        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        print(prediction)
        # test_accuracy(prediction, y)

    # print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    pre_lis = prediction.numpy()

    detected_lis = []
    for i in pre_lis:
        i = int(i)
        if i == 0:
            detected_lis.append(False)
        else:
            detected_lis.append(True)

    # tf.stack([y,prediction],axis=1)

    Detect = pd.Series(detected_lis)

    print(Final_test_data[Detect])

    test_output_box.config(state="normal")
    test_output_box.delete('1.0', END)
    test_output_box.insert(INSERT, Final_test_data[Detect])
    test_output_box.config(state="disabled")
    print(refined_data)


# display chart by prediction
def test_visual_chart():
    plt.scatter(plotcategory['ID_Details'].numpy(), plotcategory['Consumption'].numpy(), c=prediction.numpy(),
                cmap='viridis')
    plt.axis([0.0, 0.6, 0.0, 0.8])
    plt.xlabel("ID_Details")
    plt.ylabel("Consumption")
    plt.show()


#buttons in Test module

test_show_d = Button(tab_2, text="Show Test Dataset", bg="#3c3f41", fg='white', width=16, height=2, command=test_show_dataset)
test_show_d.grid(row=3, column=9, padx=70, pady=20)

# cleanup button
test_clean_b = Button(tab_2, text="Clean Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=test_clean_dataset)
test_clean_b.grid(row=5, column=3, padx=20, pady=15)

test_refined_b = Button(tab_2, text="Refined Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=test_show_refined)
test_refined_b.grid(row=7, column=3, padx=20, pady=15)

# detect button
detect_b = Button(tab_2, text="Detect Anomaly", bg="#3c3f41", fg='white', width=12, height=2, command=detect_test)
detect_b.grid(row=9, column=3, padx=20, pady=15)

# Visualise anomaly button
test_visual_b = Button(tab_2, text="Visualize", bg="#3c3f41", fg='white', width=12, height=2, command=test_visual_chart)
test_visual_b.grid(row=11, column=3, padx=20, pady=15)


# output box
global test_output_box
test_output_box = Text(tab_2, height=20, width=66,  background='#313335', foreground='white')
test_output_box.insert(INSERT, "Output Box")
test_output_box.config(state="disabled", wrap='none')
test_output_box.grid(row=5, column=6, columnspan=6, rowspan=9,pady=0)

# adding scroll bar on output window
t_scrolly = Scrollbar(tab_2,  background='#313335')
t_scrolly.grid(row=13, column=12, rowspan=2)

t_scrollx = Scrollbar(tab_2, orient='horizontal', background='#313335')
t_scrollx.grid(row=14, column=10, columnspan=3, padx=17)

t_scrolly.config(command=test_output_box.yview)
t_scrollx.config(command=test_output_box.xview)

test_output_box.config(xscrollcommand=t_scrollx.set, yscrollcommand=t_scrolly.set)

root.mainloop()







