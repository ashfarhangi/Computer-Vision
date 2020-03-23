#!/usr/bin/env python
# coding: utf-8

# ## Image to Frames

# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import io
import base64
from IPython.display import HTML
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
FOLDER = ".//data/"
plt.rcParams['figure.figsize'] = 12, 8


# In[2]:


#Testing 
cap = cv2.VideoCapture('D://UCF-101//YoYo//v_YoYo_g03_c04.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


# In[3]:


with open(FOLDER+"train1.txt") as f:
    content = f.readlines()
train = [x.strip().split(" ") for x in content] 
with open(FOLDER+"/train_test//validation1.txt") as f:
    content = f.readlines()
validation = [x.strip().split(" ") for x in content] 

with open(FOLDER+"/train_test//trainlist01.txt") as f:
    content = f.readlines()
all_data = [x.strip().split(" ") for x in content] 

with open(FOLDER+"train_test/sample.txt") as f:
    content = f.readlines()
sample = [x.strip().split(" ") for x in content] 


# In[4]:


(sample)


# In[24]:


filelist = []
for video,label in all_data:
    category = video.split("/")[0]
    filename = video.split("/")[1].split(".avi")[0]    
    directory_name = FOLDER + "UCF101_images/" + category 

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    # The directroy needs to be changed according to the directory of UCF 101 Video Files
    video_file = "D://" + "UCF-101//" + video.split("/")[0]+ "//"+video.split("/")[1]
    print(video_file)
    cap = cv2.VideoCapture(video_file)
    counter = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
#             1 second is 24 frames. Therefore we are spliting the video to 1 second jpg
        if (frameId % 24) == 0:
            f_ = directory_name + "/" + filename + "_" + str(counter) + ".jpg"
            cv2.imwrite(f_, frame)
            counter += 1
    cap.release()
    filelist.append([category + "/" + filename, counter, label])


# ## Validation

# In[25]:


np.savetxt('data/all_images1.txt', (filelist), fmt=['%s', '%s', '%s'], delimiter=' ')


# In[26]:


counter = {i:0 for i in range(101)}
train_counter = {i:0 for i in range(100)}
val_counter = {i:0 for i in range(100)}

with open("data/all_images1.txt") as f:
    for line in f:
        label = int(line.split(" ")[2])
        counter[label-1] += 1        


# In[ ]:





# In[27]:


trainlist = []
validlist = []
with open("data/all_images1.txt") as f:
    for line in f:
        label = int(line.split(" ")[2])
        if label != 37:
            if label > 37:
#                 print(label)
                label -= 1
            filename = line.split(" ")[0] + " " + line.split(" ")[1]
            train_or_val = np.random.rand()
            if train_or_val < 0.21:
                val_counter[label-1] += 1            
                validlist.append([filename, label-1])
            else:
                train_counter[label-1] += 1
                trainlist.append([filename, label-1])


# In[28]:


np.savetxt('data/train1.txt', (trainlist), fmt=['%s', '%s'], delimiter=' ')
np.savetxt('data/val1.txt', (validlist), fmt=['%s', '%s'], delimiter=' ')


# In[55]:


plt.subplot(1,2,1),plt.bar(train_counter.keys(), train_counter.values(), 1, color='b',label = 'Train')
plt.subplot(1,2,2),plt.bar(val_counter.keys(), val_counter.values(), 1, color='g',label ='Validation')
plt.legend()
# plt.xlabel('S')
plt.show()


# In[56]:


for i in train_counter:
    if i < 36:
        j = i - 1
        print (train_counter[i]+val_counter[i]/float(counter[i]))
    if i > 35:
        j = i + 1
        print (train_counter[i]+val_counter[i]/float(counter[j]))


# In[57]:


train_counter[35]

