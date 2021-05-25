#Creating GUI with tkinter
import tkinter
from tkinter import *
from prediction import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = make_prediction(msg)
        new_res =  listToString(res)
        ChatLog.insert(END, "Bot: " + new_res + '\n\n')
        IntentBox.insert(END, new_res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        if(str1 == ""):
            str1 += str(ele)
        else:
            str1 += ", " + str(ele)

        # return string
    return str1

base = Tk()
base.title("Unseen Multi-Intent Detection")
base.geometry("400x700")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

#Create the box to output intent
intent_label = Label(base, text="Identified Intents:", anchor='w', font="Arial")
IntentBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
scrollbar2 = Scrollbar(base, command=IntentBox.yview, cursor="heart")
IntentBox['yscrollcommand'] = scrollbar2.set
intent_label.place(x=6, y=501, height=30)
scrollbar.place(x=376,y=531, height=150)
IntentBox.place(x=6, y=531, height=150, width=370)

base.mainloop()