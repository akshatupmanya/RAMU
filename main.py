from backloader import mainLoader

Backdata=mainLoader()

while True:
    print("Enter Your options-->")
    print("1.Document Scanner")
    print("2.speech to text")
    print("3.text to speech")
    print("4.exit")
    opt=int(input("Enter Your Input:"))

    if opt==1:
        data=Backdata.document_loader()
        print(data)
    elif opt==2:
        data=Backdata.speech_totext()
        print(data)
    elif opt==3:
        data=Backdata.speech_totext()
    elif opt==4:
        break
    else:
        print("Error occured... wrong input")


