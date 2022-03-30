import tkinter as tk
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import easygui
import matplotlib.patches as pc
import time
window = tk.Tk()
window.title('自走車')
window.geometry('300x450')
window.configure(background='white')
l1 = tk.Label(window, text = "迭代次數",font=('Arial', 10),bg = "white", width = 12, height = 2 )
l1.place(x=10,y=20)
l2 = tk.Label(window, text = "族群大小",font=('Arial', 10),bg = "white", width = 12, height = 2 )
l2.place(x=10,y=80)
l3 = tk.Label(window, text = "突變機率",font=('Arial', 10),bg = "white", width = 12, height = 2 )
l3.place(x=10,y=140)
l4 = tk.Label(window, text = "交配機率",font=('Arial', 10),bg = "white", width = 12, height = 2 )
l4.place(x=10,y=200)
l5 = tk.Label(window, text = "網路J值",font=('Arial', 10),bg = "white", width = 12, height = 2 )
l5.place(x=10,y=260)
var1 = tk.StringVar()
var1.set("尚未訓練完成")
l6 = tk.Label(window, textvariable=var1,font=('Arial', 10),fg = "red",bg = "white", width = 12, height = 2)
l6.place(x=100,y=400)
e1 = tk.Entry(window, show=None,width = 12)   
e2 = tk.Entry(window, show=None,width = 12)  # 顯示成明文形式
e3 = tk.Entry(window, show=None,width = 12)   
e4 = tk.Entry(window, show=None,width = 12)  # 顯示成明文形式
e5 = tk.Entry(window, show=None,width = 12)   
e1.place(x=130,y=30)
e2.place(x=130,y=90)
e3.place(x=130,y=150)
e4.place(x=130,y=210)
e5.place(x=130,y=270)
flag1 = 0
loop_time = 0
family_size = 0
mutation_rate = 0
crossover_rate = 0
internet_j = 0
gene_size = 0
family_list = []
lists = []
lists2 = []
ans = [0]*gene_size
E_number = [0]*family_size
x = []
y = []
E_min = 1000
degree = 0

figure, axes = plt.subplots()
origin = [0,0]
head = [0,0]
headleft45 = [0,0]
headright45 = [0,0]
lenhead = 100
lenheadleft45 = 100
lenheadright45 = 100
len_of_line = [0,0,0]
temp = [0,0]

def rotatecounterclockwise(origin,head,angle):
    global temp
    temp[0] = (head[0]-origin[0])*math.cos(math.radians(angle)) - (head[1]-origin[1])*math.sin(math.radians(angle))+origin[0]
    temp[1] = (head[0]-origin[0])*math.sin(math.radians(angle)) + (head[1]-origin[1])*math.cos(math.radians(angle))+origin[1]
def distance(origin,head):
    global lists
    lenhead = 100
    x1 = origin[0]
    y1 = origin[1]
    x2 = head[0]
    y2 = head[1]
    if(x1 == x2):
        k1 = None
        b1=0
    else:
        k1 = (y2-y1)*1.0/(x2-x1)
        b1=y1*1.0-x1*k1*1.0
    X=0
    Y=0
    minx=0
    miny=0
    for i in range(3,len(lists)-1):
        flag = 0
        x3 = float(lists[i][0])
        x4 = float(lists[i+1][0])
        y3 = float(lists[i][1])
        y4 = float(lists[i+1][1])
        if(x3 == x4):
            flag = 1
        if(x4-x3) == 0:
            k2 = None
            b2 = 0
        else:
            k2=(y4-y3)*1.0/(x4-x3)
            b2=y3*1.0-x3*k2*1.0
        if(k1!=k2):
            if k2==None:
                X=x3
                Y=k1*X*1.0+b1*1.0
            elif k1==None:
                X=x1
                Y=k2*X*1.0+b2*1.0
            else:
                X=(b2-b1)*1.0/(k1-k2)
                Y=k1*X*1.0+b1*1.0
            if(flag == 0):
                if(X<=max(x3,x4) and X>=min(x3,x4)):
                    if (head[0] - X)**2+(head[1]-Y)**2 < (origin[0] - X)**2+(origin[1]-Y)**2:
                        if(((origin[0] - X)**2+(origin[1]-Y)**2)**0.5 < lenhead):
                            minx = X
                            miny = Y
                            lenhead = min(lenhead,((origin[0] - X)**2+(origin[1]-Y)**2)**0.5)
            if (flag == 1):
                if Y>=min(y3,y4) and Y<=max(y3,y4):
                    if (head[0] - X)**2+(head[1]-Y)**2 < (origin[0] - X)**2+(origin[1]-Y)**2:
                        if(((origin[0] - X)**2+(origin[1]-Y)**2)**0.5 < lenhead):
                            minx = X
                            miny = Y
                            lenhead = min(lenhead,((origin[0] - X)**2+(origin[1]-Y)**2)**0.5)

    # plt.scatter(minx,miny,color = "g")
    return lenhead
def compute_E():
    global degree,family_size,E_number,family_list
    for i in range(family_size):
        for j in range(len(lists)):
            fx = family_list[i][0]
            start_pointP = internet_j+1
            start_pointQ = degree*internet_j+1
            start_pointW = 1
            for k in range(internet_j):
                p = 0
                for l in range(degree-1):
                    p += (lists[j][l]-family_list[i][start_pointP+l])**2
                p = -1*p/(2*(family_list[i][start_pointQ])**2)
                p = math.exp(p)
                start_pointP += degree-1
                start_pointQ += 1
                fx += family_list[i][start_pointW]*p
                start_pointW += 1
            E_number[i] += (1/2)*(lists[j][degree-1]-fx)**2

def gogo():
    global lists ,loop_time ,family_size ,mutation_rate ,crossover_rate ,internet_j ,gene_size ,family_list ,E_number ,ans ,flag1 ,var1,E_min,degree
    lists = []
    train_par = open('train_par.txt', 'w')
    start = time.process_time()
    path = easygui.fileopenbox()
    file = open(path, mode="r")
    for line in file:
        list = line.split()
        lists.append(list)
    degree = len(lists[0])
    for i in range(len(lists)):
        for j in range(degree):
            if(j!=degree-1):
                lists[i][j] = float(lists[i][j])/40-1
            else:
                lists[i][j] = float(lists[i][j])/40
    loop_time = int(e1.get())
    family_size = int(e2.get())
    mutation_rate = float(e3.get())
    crossover_rate = float(e4.get())
    internet_j = int(e5.get())
    gene_size = 1 + (degree+1)*internet_j
    family_list = []
    E_min = 10000
    for i in range(family_size):
        temp5 = []
        for j in range(gene_size):
            if(j < 1 + degree*internet_j):
                temp5 += [random.uniform(-1,1)]
            else:
                temp5 += [random.uniform(0,1)]
        family_list.append(temp5)
    
    for l in range(loop_time):
        E_number = [0]*family_size
        compute_E()
        # print(E_number)
        mini = 0
        for i in range(family_size):
            if(E_number[mini] > E_number[i]):
                mini = i
        if(E_number[mini] < E_min):
            E_min = E_number[mini]
            ans = [0]*gene_size
            for i in range(gene_size):
                ans[i] = family_list[mini][i]

        E_total = 0
        for i in range(len(E_number)):
            E_total += E_number[i]
        for i in range(len(E_number)):
            E_number[i] /= E_total
        for i in range(len(E_number)):
            E_number[i] = 1 / E_number[i]
        E_total = 0
        for i in range(len(E_number)):
            E_total += E_number[i]
        for i in range(len(E_number)):
            E_number[i] /= E_total
        
        for i in range(1,len(E_number)):
            E_number[i] += E_number[i-1]
        ##複製
        if(random.uniform(0,1)<=0.5):
            compare_family = []
            for i in range(family_size):
                compare_family += [family_list[i]]
            for i in range(family_size):
                temp1 = random.uniform(0,1)
                for j in range(family_size):
                    if(E_number[j] >= temp1):
                        for k in range(gene_size):
                            family_list[i][k] = compare_family[j][k]
                        break
        else:
            compare_family = []
            for i in range(family_size):
                compare_family += [family_list[i]]
            for i in range(family_size):
                temp1 = random.randrange(0, family_size-1)
                temp6 = random.randrange(0, family_size-1)
                if(E_number[temp1] >= E_number[temp6]):
                    for k in range(gene_size):
                        family_list[i][k] = compare_family[temp6][k]
                else:
                    for k in range(gene_size):
                        family_list[i][k] = compare_family[temp1][k]
        #交配
        CM_rate = 0.1
        for i in range(family_size):
            if(i%2==0 and i!=family_size-1):
                temp1 = random.uniform(0,1)
                if(temp1<=crossover_rate):
                    for j in range(gene_size):
                        temp2 = family_list[i][j]
                        temp3 = family_list[i+1][j]
                        family_list[i][j] += CM_rate*(temp2-temp3) 
                        family_list[i+1][j] -= CM_rate*(temp2-temp3)
        #突變
        for i in range(family_size):
            temp1 = random.uniform(0,1)
            if temp1 <= mutation_rate:
                temp4 = random.uniform(-1,1)
                for j in range(gene_size):
                    family_list[i][j] += 0.1*temp4
    compute_E()
    mini = 0
    for i in range(family_size):
        if(E_number[mini] > E_number[i]):
            mini = i
    if(E_number[mini] < E_min):
        E_min = E_number[mini]
        ans = [0]*gene_size
        for i in range(gene_size):
            ans[i] = family_list[mini][i]
    flag1 = 1
    var1.set("訓練完成")
    end = time.process_time()  
    print("執行時間：%f 秒" % (end - start))
    for i in range(gene_size):
        train_par.write(str(ans[i]))
        train_par.write(" ")
    ####上方訓練完成ans放網路參數

def put_map():
    global lists ,ans ,flag1 ,x,y,origin,head,headleft45,headright45,lenhead,lenheadleft45,lenheadright45,len_of_line,temp,axes,figure,internet_j
    if(flag1 == 1):
        lists = []
        path = easygui.fileopenbox()
        file = open(path, mode="r")
        for line in file:
            line = line.strip('\n')
            list = line.split(",")
            lists.append(list)
        x = []
        y = []
        origin = [int(lists[0][0]),int(lists[0][1])]
        head = [origin[0]+4,origin[1]]
        headleft45 = [0,0]
        headright45 = [0,0]
        lenhead = 100
        lenheadleft45 = 100
        lenheadright45 = 100
        len_of_line = [0,0,0]
        temp = [0,0]
        #################################
        change_angle = 0
        t = 100
        for i in range(3,len(lists)-1):
            x = []
            y = []
            x = x + [float(lists[i][0])]
            x = x + [float(lists[i+1][0])]
            y = y + [float(lists[i][1])]
            y = y + [float(lists[i+1][1])]
            plt.scatter(x,y,color = "b")
            plt.plot(x,y,color = "r")       
        endline1 = [float(lists[1][0]),float(lists[1][1])]
        endline2 = [float(lists[2][0]),float(lists[2][1])]
        axes.add_patch(pc.Rectangle(  (min(endline1[0],endline2[0]), min(endline1[1],endline2[1])), abs(endline1[0]-endline2[0]), abs(endline1[1]-endline2[1]),  color='#B2B2B2'  ))
        while True:
            flag_pause = 0
            for i in range(3,len(lists)-1):
                flag = 0
                line_point1 = [float(lists[i][0]),float(lists[i][1])]
                line_point2 = [float(lists[i+1][0]),float(lists[i+1][1])]
                if(line_point1[0] == line_point2[0]):
                    flag = 1
                point = [origin[0],origin[1]]
                A = line_point2[1] - line_point1[1]
                B = line_point1[0] - line_point2[0]
                C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]
                distance_origin = abs(A * point[0] + B * point[1] + C) / (math.sqrt(A**2 + B**2))
                if(distance_origin < 3):
                    if(flag == 0):
                        if(point[0]<=max(line_point1[0],line_point2[0]) and point[0]>=min(line_point2[0],line_point1[0])):
                            flag_pause = 1
                    else:
                        if(point[1]<=max(line_point1[1],line_point2[1]) and point[1]>=min(line_point2[1],line_point1[1])):
                            flag_pause = 1
            if(flag_pause == 1):
                break
            ##############################################################
            ###########################################################前方向
            if t == 100: 
                angle = float(lists[0][2])
                rotatecounterclockwise(origin,head,angle)
            else:
                head[0] = origin[0]+4
                head[1] = origin[1]
                rotatecounterclockwise(origin,head,angle)
            for i in range(2):
                head[i] = (temp[i])
            x = []
            y = []
            x = x + [(head[0])]
            x = x + [(origin[0])]
            y = y + [(head[1])]
            y = y + [(origin[1])]
            len_of_line[1]= distance(origin,head)
            ###########################################################前方向
            ###########################################################左方向 
            rotatecounterclockwise(origin,head,45)
            for i in range(2):
                headleft45[i] = temp[i]
            x = []
            y = []
            x = x + [(headleft45[0])]
            x = x + [(origin[0])]
            y = y + [(headleft45[1])]
            y = y + [(origin[1])]
            len_of_line[0] = distance(origin,headleft45)
            ###########################################################左方向
            ###########################################################右方向
            rotatecounterclockwise(origin,head,-45)
            for i in range(2):
                headright45[i] = temp[i]
            x = []
            y = []
            x = x + [(headright45[0])]
            x = x + [(origin[0])]
            y = y + [(headright45[1])]
            y = y + [(origin[1])]
            len_of_line[2] = distance(origin,headright45)
            ###########################################################右方向
            ###########################################################角度轉變
            fx = ans[0]
            start_pointP = internet_j+1
            start_pointQ = degree*internet_j+1
            start_pointW = 1
            # print((len_of_line[1]/40)-1," ",(len_of_line[2]/40)-1," ",(len_of_line[0]/40)-1)
            for k in range(internet_j):
                p = 0
                if degree == 4:
                    p += ((len_of_line[1]/40)-1-ans[start_pointP])**2
                    p += ((len_of_line[2]/40)-1-ans[start_pointP+1])**2
                    p += ((len_of_line[0]/40)-1-ans[start_pointP+2])**2
                    start_pointP += 3
                elif degree == 6:
                    p += ((origin[0]/40)-1-ans[start_pointP])**2
                    p += ((origin[1]/40)-1-ans[start_pointP+1])**2
                    p += ((len_of_line[1]/40)-1-ans[start_pointP+2])**2
                    p += ((len_of_line[2]/40)-1-ans[start_pointP+3])**2
                    p += ((len_of_line[0]/40)-1-ans[start_pointP+4])**2
                    start_pointP += 5
                p = -1*p/(2*(ans[start_pointQ])**2)
                p = math.exp(p)
                start_pointQ += 1
                fx += ans[start_pointW]*p
                start_pointW += 1
            
            change_angle = fx*40
            ###########################################################角度轉變
            ox = origin[0]
            oy = origin[1]
            origin[0] += math.cos(math.radians(angle+change_angle)) + math.sin(math.radians(angle))*math.sin(math.radians(change_angle))
            origin[1] += math.sin(math.radians(angle+change_angle)) - math.sin(math.radians(change_angle))*math.cos(math.radians(angle))
            angle -= math.degrees(math.asin(2*math.radians(change_angle)/3))

            x = []
            y = []
            x = x + [ox]        
            x = x + [origin[0]]
            y = y + [oy]
            y = y + [origin[1]]
            axes.plot(x,y,color = "r")
            t -= 1
            if(origin[0]<= max(endline1[0],endline2[0]) and origin[0]>= min(endline1[0],endline2[0])):
                if(origin[1]<= max(endline1[1],endline2[1]) and origin[1]>= min(endline1[1],endline2[1])):
                    plt.text(35,20,"successful",color = "r")
                    break
        plt.show()
            
    


b1 = tk.Button(window, text='開始訓練', font=('Arial', 12), width=10, height=2,command = gogo)
b1.place(x=40,y=320)
b2 = tk.Button(window, text='放上地圖', font=('Arial', 12), width=10, height=2,command = put_map)
b2.place(x=160,y=320)
window.mainloop()

