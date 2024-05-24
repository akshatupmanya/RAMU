from backloader import mainLoader

Backdata = mainLoader()

while True:
    print("Enter Your options-->")
    print("1.Document Scanner(Requires Camera)")
    print("2.speech to text")
    print("3.Exit")
    opt = int(input("Enter Your Input:"))

    if opt == 1:
        data = Backdata.document_loader()
        print(data)
    elif opt == 2:
        data = Backdata.speech_totext()
        print(data)
    elif opt == 3:
        break
    else:
        print("Error occured... wrong input")
