import socket
from tkinter import *
import datetime
from PIL import ImageTk, Image

root = Tk()
#setting the title, size and color of the window
root.title("Smart Grid Sub-Station Request Generator")
root.geometry('480x480')
root.resizable(0, 0)
root.configure(background='#2b2b2b')

'''pic = Frame(root)
image_t1 = ImageTk.PhotoImage(Image.open("D:\Photoshop\Phoshop edits\project bg1png.png"))
pic.create_image(0, 0, anchor=NW, image=image_t1)

pic.pack()'''

# INCLUDING CLIENT DETAILS IN SEPARATE VARIABLES
global client_ID, client_IP, client_details, day
client_ID = "10103"
client_IP = "193.101.77.61"
date = datetime.datetime.now()
DATE = date.strftime("%d/%m/%Y")
DATE = str(DATE)
DATE = DATE[0:10]
day = date.strftime("%w")
day = str(int(day)+1)

# client socket program
global ADDR, FORMAT, client, SERVER, output_box, cons_entry
client_details = [client_ID, client_IP]
HEADER = 64
# setting the port number and IP of the server to establish connection
PORT = 9999
# give the IP of the server
SERVER = "192.168.137.1"
ADDR = (SERVER, PORT)

FORMAT = "utf-8"
dis_msg = '!DISCONNECTED'


# connecting to the server
def connect():
    global client
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(ADDR)
        #ser_msg = client.recv(1024)
        output_box.config(state="normal")
        output_box.delete('1.0', END)
        output_box.insert(INSERT, "Successfully connected to the server")
        output_box.config(state="disabled")
        connect_button.config(state="disabled")
        #print(ser_msg)
        send_button.config(state='normal')
    except Exception as e:
        print("Server is not running",e)
        output_box.config(state="normal")
        output_box.delete('1.0', END)
        output_box.insert(INSERT, "Server is not running")
        output_box.config(state="disabled")


# sending request to the server
def send_function():
    global client, cons_entry, client_details, output_box
    value = cons_entry.get()
    if value:
        try:
            value  = float(value)
            client_details.append(str(value))
            client_details.append(day)

            msg = "-".join(client_details)

            message = msg.encode(FORMAT)
            #msg_l = len(message)
            #send_len = str(msg_l).encode(FORMAT)
            #send_len += b' ' * (HEADER - len(send_len))
            #client.send(send_len)
            client.send(message)
            send_button.config(state="disabled")

            output_box.config(state="normal")
            output_box.delete('1.0', END)
            output_box.insert(INSERT, "Request has been sent... please wait....")
            output_box.config(state="disabled")
            print("Request has been sent... please wait....")

            response_msg = client.recv(1024).decode(FORMAT)
            response_lis = response_msg.split('-')
            output_box.config(state="normal")
            output_box.delete('1.0', END)
            output_box.insert(INSERT, f"{response_lis[0]} \n \n{response_lis[1]}")
            output_box.config(state="disabled")


        except Exception as e:
            output_box.config(state="normal")
            output_box.delete('1.0', END)
            output_box.insert(INSERT, "Enter the consumption rate in number")
            output_box.config(state="disabled")
            print("Enter the consumption rate in number", e)
    else:
        output_box.config(state="normal")
        output_box.delete('1.0', END)
        output_box.insert(INSERT, "Enter the consumption rate in number")
        output_box.config(state="disabled")
        print("Enter the consumption rate in number")



# displaying the client details through labels and output box
ID_label = Label(root, text=f"SUB-STATION ID : {client_ID}", bg="#313335", fg='white', width=25, height=2)
ID_label.grid(row=3, column=3, columnspan=2, padx=30, pady=5)

IP_label = Label(root, text=f"SUB-STATION IP : {client_IP}", bg="#313335", fg='white', width=25, height=2)
IP_label.grid(row=4, column=3, columnspan=2, padx=20, pady=5)

date_label = Label(root, text=f"CURRENT DATE : {DATE}", bg="#313335", fg='white', width=25, height=2)
date_label.grid(row=5, column=3, columnspan=2, padx=20, pady=5)

connect_button = Button(root, text="Connect to Server", bg="#3c3f41", fg='white', command=connect, width=25, height=4)
connect_button.grid(row=4, column=5, rowspan=2)

output_box = Text(root, height=10, width=55,  background='#313335', foreground='white')
output_box.insert(INSERT, "Connect to the server.....")
output_box.config(state="disabled")
output_box.grid(row=7, column=3, columnspan=5, rowspan=8,padx=17, pady=20)

cons_label = Label(root, text="Consumption : ", bg="#313335", fg='white', width=25, height=2)
cons_label.grid(row=15,column=3, columnspan=2, padx=5, pady=5)

cons_entry = Entry(root, text="0.3", width=25)
cons_entry.grid(row=15, column=5, columnspan=2, padx=5, pady=5)

send_button = Button(root, text="Send Request", bg="#3c3f41", fg='white', width=30, height=2, command=send_function)
send_button.grid(row=16, column=3, columnspan=4, padx=20, pady=15)
send_button.config(state='disabled')

value = cons_entry.get()


s = "-".join(client_details)


root.mainloop()