import datetime
import time
import Recognise
import threading

current = datetime.datetime
start=[16,43]
duration = 2
rec = Recognise.Recognition()


while True:
    print('hour is '+str(current.now().time().hour))
    print('minute is '+str(current.now().time().minute))
    time.sleep(1)
    if current.now().time().hour==start[0] and current.now().time().minute==start[1]:
        print("It's time")
        looking = threading.Thread(target=rec.schedcallrecognise, args=(['Nicholas','Martia'],),daemon = True)
        looking.start()
        while current.now().time().minute!=start[1]+duration and rec.killthread == False:
            pass
        rec.killthread = True
        
        
    