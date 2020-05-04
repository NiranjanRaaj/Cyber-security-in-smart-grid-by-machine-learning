import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import socket
import datetime
import threading
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
root.resizable(0, 0)
root.configure(background='#2b2b2b')

#setting tab layout
tab = ttk.Notebook(root)


image_t1 = ImageTk.PhotoImage(Image.open("D:\Photoshop\Phoshop edits\project bg1png.png"))
image_t2 = ImageTk.PhotoImage(Image.open("D:\Photoshop\Phoshop edits\project bg2png.png"))
image_t3 = ImageTk.PhotoImage(Image.open("D:\Photoshop\Phoshop edits\project bg3png.png"))

tab_1 = Canvas(tab)
tab_1.create_image(0, 0, anchor=NW, image=image_t1)
tab.add(tab_1, text="Train Module")

tab_2 = Canvas(tab, background='#313335')
tab_2.create_image(0, 0, anchor=NW, image=image_t2)
tab.add(tab_2, text="Test Module")

tab_3 = Canvas(tab, background='#2b2b2b')
tab_3.create_image(0, 0, anchor=NW, image=image_t3)
tab.add(tab_3, text="Real-Time Detection")

tab.pack(expand=2, fill='both')

'''************************************************* TRAIN MODULE ***************************************************'''
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
    if traindata_loc:
        output_box.config(state="normal")
        output_box.delete('1.0', END)
        e_ent.insert(INSERT, traindata_loc)
        e_ent.config(state="disabled")
        output_box.insert(INSERT, "DATASET SUCCESSFULLY LOADED")
        output_box.config(state="disabled")

        show_d.config(state='normal')
        clean_b.config(state='normal')


dialog_b = Button(tab_1, text=". . .",  bg="#3c3f41", fg='white', width=12, height=1, command=open_file)
dialog_b.grid(row=3, column=6, padx=50, pady=30)

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

    train_b.config(state='normal')
    refined_b.config(state='normal')
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
    output_box.insert(INSERT, "Please wait... Training process going on")
    output_box.config(state="disabled")
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

    output_box.config(state="normal")
    output_box.delete('1.0', END)
    for l in accuracy_lis:
        output_box.insert(INSERT, l + '\n')
    output_box.config(state="disabled")
    visual_b.config(state='normal')
    test_dialog_b.config(state='normal')
    start_button.config(state="normal")


def start_train():
    t1 = threading.Thread(target=train_ml)
    t1.start()


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
show_d.config(state='disabled')

# cleanup button
clean_b = Button(tab_1, text="Clean Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=clean_dataset)
clean_b.grid(row=5, column=9, padx=100, pady=15)
clean_b.config(state='disabled')

# Show refined dataset button
refined_b = Button(tab_1, text="Refined Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=show_refined)
refined_b.grid(row=7, column=9, padx=100, pady=15)
refined_b.config(state='disabled')

# train button
train_b = Button(tab_1, text="Train Machine", bg="#3c3f41", fg='white', width=12, height=2, command=start_train)
train_b.grid(row=9, column=9, padx=100, pady=15)
train_b.config(state='disabled')

# visual button
visual_b = Button(tab_1, text="Visualize", bg="#3c3f41", fg='white', width=12, height=2, command=visual_graph)
visual_b.grid(row=11, column=9, padx=100, pady=15)
visual_b.config(state='disabled')

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


'''------------------------------------------------ TEST MODULE ----------------------------------------------------'''
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
    if testdata_loc:

        test_output_box.config(state="normal")
        test_output_box.delete('1.0', END)
        test_e_ent.insert(INSERT, testdata_loc)
        test_e_ent.config(state="disabled")
        test_output_box.insert(INSERT, "TESTING DATASET SUCCESSFULLY LOADED")
        test_output_box.config(state="disabled")
        test_show_d.config(state='normal')
        test_clean_b.config(state='normal')


test_dialog_b = Button(tab_2, text=". . .",  bg="#3c3f41", fg='white', width=12, height=1, command=test_open_file)
test_dialog_b.grid(row=3, column=6, padx=50,pady=30)
test_dialog_b.config(state='disabled')


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
    global val, val_ID, dic, ID_dic, IP_col, ID_col, features, labels
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
    test_refined_b.config(state='normal')
    detect_b.config(state='normal')


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
    test_visual_b.config(state='normal')


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
test_show_d.config(state='disabled')

# cleanup button
test_clean_b = Button(tab_2, text="Clean Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=test_clean_dataset)
test_clean_b.grid(row=5, column=3, padx=20, pady=15)
test_clean_b.config(state='disabled')

test_refined_b = Button(tab_2, text="Refined Dataset", bg="#3c3f41", fg='white', width=12, height=2, command=test_show_refined)
test_refined_b.grid(row=7, column=3, padx=20, pady=15)
test_refined_b.config(state='disabled')

# detect button
detect_b = Button(tab_2, text="Detect Anomaly", bg="#3c3f41", fg='white', width=12, height=2, command=detect_test)
detect_b.grid(row=9, column=3, padx=20, pady=15)
detect_b.config(state='disabled')

# Visualise anomaly button
test_visual_b = Button(tab_2, text="Visualize", bg="#3c3f41", fg='white', width=12, height=2, command=test_visual_chart)
test_visual_b.grid(row=11, column=3, padx=20, pady=15)
test_visual_b.config(state='disabled')


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


'''++++++++++++++++++++++++++++++++++++++++++++++ REAL-TIME DETECTION MODULE ++++++++++++++++++++++++++++++++++++++++'''

global DAY, run, connection, conn, stop, server
global request_rev, request_grt, anomaly_det, total_request,previous_request, new_request
request_rev, request_grt, anomaly_det = 0, 0, 0
previous_request = []

stop = False
# setting the port number and IP of the server
DATE = datetime.datetime.now()
current_d = (DATE.strftime('%w'))
DAY = int(current_d) + 1
PORT = 9999

HEADER = 64
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
dis_msg ="!DISCONNECTED"

run = True
connection = True
# making the server by binding and making the socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setblocking(0)
server.bind(ADDR)

global client_ID, client_IP, consumption, c_day


# function for handling the client, get the meg from client
def handle_client(conn, addr):
    global connection, client_ID, client_IP, consumption, c_day, converted_ID, converted_IP, model, type
    global val, val_ID, dic, ID_dic, IP_col, ID_col, request_grt, anomaly_det, request_rev, total_request, previous_request, new_request

    connection = True
    print(f"NEW CONNECTION {addr} connected")
    rt_output_box.config(state='normal')
    rt_output_box.delete('1.0', END)
    rt_output_box.insert(INSERT, "New connection received")
    rt_output_box.config(state="disabled", wrap='none')

    request_rev = request_rev +1
    req_text.config(state='normal')
    req_text.delete('1.0', END)
    req_text.insert(INSERT, request_rev)
    req_text.config(state="disabled", wrap='none')

    class_names = ["Normal", "Anomaly"]
    while connection:
        try:

            '''co_msg = "connected"
            co_msg = co_msg.encode(FORMAT)
            conn.send(co_msg)'''

            client_msg = conn.recv(1024).decode(FORMAT)
            msg_lis = client_msg.split('-')

            rt_output_box.config(state='normal')
            rt_output_box.insert(INSERT, "\n \n Request Details : ")
            rt_output_box.insert(INSERT, f"\n \n Sub_Station ID = {msg_lis[0]} \n Sub-station IP = {msg_lis[1]} \n Consumption rate = {msg_lis[2]}")
            rt_output_box.config(state="disabled", wrap='none')

            client_ID, client_IP, consumption, c_day = msg_lis[0], msg_lis[1], msg_lis[2], msg_lis[3]
            if len(msg_lis) > 1:

                # converting the client ID
                client_ID = int(client_ID)
                if client_ID not in ID_col:
                    # Assigning new value to ID if the ID is new
                    val_ID = val_ID + 0.1
                    new_st = client_ID
                    new_d = {new_st: round(val_ID, 1)}
                    ID_dic.update(new_d)

                converted_ID = ID_dic[(client_ID)]

                # converting the client IP
                if client_IP not in IP_col:
                    # Assigning new value to IP if the IP is new
                    val = val + 0.1
                    st = client_IP
                    d = {st: round(val, 1)}
                    dic.update(d)

                converted_IP = dic[(client_IP)]

                # converting the consumption and day
                consumption = float(consumption)
                req_day = c_day
                c_day = float(c_day)

                rt_output_box.config(state='normal')
                rt_output_box.insert(INSERT, "\n \n converted Details : ")
                rt_output_box.insert(INSERT,
                                     f"\n \n Sub_Station ID = {converted_ID} \n Sub-station IP = {converted_IP}"
                                     f" \n Consumption rate = {consumption} \n day = {c_day}")
                rt_output_box.config(state="disabled", wrap='none')

                # detecting the request with trained model
                predict_dataset = tf.convert_to_tensor([[converted_ID, converted_IP, consumption, c_day]])

                predictions = model(predict_dataset)

                for i, logits in enumerate(predictions):
                    class_idx = tf.argmax(logits).numpy()
                    p = tf.nn.softmax(logits)[class_idx]
                    type = class_names[class_idx]
                    print("Example {} prediction: {} ({:4.1f}%)".format(i, type, 100 * p))

                # appendind the request details to view the previous request
                new_request = {"Sub-Station ID": client_ID, "Sub-Station IP": client_IP, "Consumption": consumption, "Day": req_day, "Type": type}
                previous_request.append(new_request)
                total_request = pd.DataFrame(previous_request, columns=["Sub-Station ID", "Sub-Station IP", "Consumption", "Day", "Type"])

                # displaying and send response according to the request type
                if type == "Normal":
                    request_grt = request_grt + 1

                    rt_output_box.config(state='normal')
                    rt_output_box.insert(INSERT, "\n \n Request is Normal, PERMISSION GRANTED ")
                    rt_output_box.config(state="disabled", wrap='none')
                    res_msg = "PERMISSION GRANTED  - Your request is normal"

                    grant_text.config(state="normal")
                    grant_text.delete('1.0', END)
                    grant_text.insert(INSERT, request_grt)
                    grant_text.config(state="disabled", wrap='none')

                    res_msg = res_msg.encode(FORMAT)
                    conn.send(res_msg)

                else:
                    anomaly_det = anomaly_det + 1

                    rt_output_box.config(state='normal')
                    rt_output_box.insert(INSERT, "\n \n Request is Anomaly, PERMISSION NOT GRANTED ")
                    rt_output_box.config(state="disabled", wrap='none')
                    res_msg = "PERMISSION NOT GRANTED... - Anomaly detected in your request"

                    ad_text.config(state="normal")
                    ad_text.delete('1.0', END)
                    ad_text.insert(INSERT, anomaly_det)
                    ad_text.config(state="disabled", wrap='none')

                    res_msg = res_msg.encode(FORMAT)
                    conn.send(res_msg)

                print(msg_lis)
                conn.close()
                connection = False
        except:
            continue
        '''while connection:
            if client_msg == dis_msg:
                connection = False'''

    conn.close()


# function for starting the server
def start_server():
    server.listen(5)
    print("SERVER IS STARTING............")
    print(f"SERVER IP ADDRESS IS {SERVER}")

    start_button.config(state='disabled')
    rt_output_box.config(state='normal')
    rt_output_box.delete('1.0', END)
    rt_output_box.insert(INSERT, "SERVER STARTED..... WAITING FOR CONNECTION")
    rt_output_box.config(state="disabled", wrap='none')

    end_button.config(state="normal")

    global run, stop
    stop = False
    while run:

        global request_rev

        if stop == True:
            print("server stopped.......")
            rt_output_box.config(state='normal')
            rt_output_box.delete('1.0', END)
            rt_output_box.insert(INSERT, "SERVER STOPPED")
            rt_output_box.config(state="disabled", wrap='none')
            break
        try:
            conn, addr = server.accept()

            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()

            print(f"\n ACTIVE CONNECTION : {threading.activeCount() -1}")
        except :
            continue


def stop_server():
    global stop, run, server, connection, t
    start_button.config(state='normal')
    #run = False
    stop = True
    connection = False
    print("value updated")


def start_runtime():
    global t
    t = threading.Thread(target=start_server)
    t.start()


'''def stop_runtime():
    server.stop()
    #t = threading.Thread(target=server.stop)
    #t.start()'''

# function for showing the previous request
def show_pre_request():
    root2 = Tk()
    root2.title("PREVIOUS REQUEST")
    root2.geometry('800x480')
    root2.resizable(0, 0)
    root2.configure(background='#2b2b2b')

    global request_output_box, total_request
    if previous_request:
        request_output_box = Text(root2, height=26, width=92, background='#313335', foreground='white')
        request_output_box.insert(INSERT, total_request)
        request_output_box.config(state="disabled", wrap='none')
        request_output_box.grid(row=5, column=3, columnspan=5, rowspan=8, pady=30, padx=30)
    else:
        request_output_box = Text(root2, height=26, width=92, background='#313335', foreground='white')
        request_output_box.insert(INSERT, "No Previous Request")
        request_output_box.config(state="disabled", wrap='none')
        request_output_box.grid(row=5, column=3, columnspan=5, rowspan=8, pady=30, padx=30)


    root2.mainloop()


# request received label
req_label = Label(tab_3, text="Request Received : ", bg="#3c3f41", fg='white', width=15, height=2)
req_label.grid(row=3, column=2, padx=20, pady=30)

# Display box of request received
req_text = Text(tab_3, width=5, height=1)
req_text.insert(INSERT, "0")
req_text.config(state="disabled", wrap='none')
req_text.grid(row=3, column=3)

# request granted label
grant_label = Label(tab_3, text="Request Granted : ", bg="#3c3f41", fg='white', width=15, height=2)
grant_label.grid(row=5, column=2, padx=20, pady=30)

# Display box of request granted
grant_text = Text(tab_3, width=5, height=1)
grant_text.insert(INSERT, "0")
grant_text.config(state="disabled", wrap='none')
grant_text.grid(row=5, column=3)

# anomaly detected label
ad_label = Label(tab_3, text="Anomaly Detected : ", bg="#3c3f41", fg='white', width=15, height=2)
ad_label.grid(row=7, column=2, padx=20, pady=30)

# Display box of anomaly detected
ad_text = Text(tab_3, width=5, height=1)
ad_text.insert(INSERT, "0")
ad_text.config(state="disabled", wrap='none')
ad_text.grid(row=7, column=3)


# setting buttons for starting and stopping the real time detection
start_button = Button(tab_3, text="START REAL-TIME", bg="#3c3f41", fg='white', width=16, height=2, command=start_runtime)
start_button.grid(row=3, column=5, padx=150, pady=20)
start_button.config(state="disabled")

end_button = Button(tab_3, text="END REAL-TIME", bg="#3c3f41", fg='white', width=16, height=2, command=stop_server)
end_button.grid(row=3, column=7, padx=0, pady=20)
end_button.config(state="disabled")

# button for displaying the previous request
pre_req_button = Button(tab_3, text="Check Previous Request", bg="#3c3f41", fg='white', width=30, height=2, command=show_pre_request)
pre_req_button.grid(row=9, column=2, padx=30, pady=20, columnspan=2)


# Output Box
global rt_output_box
rt_output_box = Text(tab_3, height=20, width=66,  background='#313335', foreground='white')
rt_output_box.insert(INSERT, "Press start run-time to start the server ")
rt_output_box.config(state="disabled", wrap='none')
rt_output_box.grid(row=5, column=5, columnspan=6, rowspan=9,pady=0)

# adding scroll bar on output window
'''rt_scrolly = Scrollbar(tab_3,  background='#313335')
rt_scrolly.grid(row=5, column=8, rowspan=9, sticky=NS)

rt_scrollx = Scrollbar(tab_3, orient='horizontal', background='#313335')
rt_scrollx.grid(row=14, column=5, columnspan=4, padx=15, sticky=EW)

rt_scrolly.config(command=rt_output_box.yview)
rt_scrollx.config(command=rt_output_box.xview)

rt_output_box.config(xscrollcommand=rt_scrollx.set, yscrollcommand=rt_scrolly.set)'''

root.mainloop()







