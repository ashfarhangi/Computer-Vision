#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
counter = {i:0 for i in range(101)}
train_counter = {i:0 for i in range(101)}
val_counter = {i:0 for i in range(101)}

with open("data/train_test/trainlist01.txt") as f:
    for line in f:
        label = int(line.split(" ")[1])
        filename = line.split(" ")[0].split("/")[1]
        counter[label-1] += 1        
#         if(val_counter[label-1] < 7):
#             val_counter[label-1] += 1
#             with open("data/validation1.txt", "a") as myfile:
#                 myfile.write(filename + " " + str(label-1)+"\n")           
#         else:
#             train_counter[label-1] += 1
# #             with open("data/train1.txt", "a") as myfile:
#                 myfile.write(filename + " " + str(label-1)+"\n")
            
#         counter[label-1] += 1


# In[12]:


plt.figure(figsize=(40, 20))
plt.subplot(1,2,1),plt.bar(train_counter.keys(), train_counter.values(), 1, color='g')
plt.subplot(1,2,2),plt.hist(val_counter.values(), 1, color='b')

plt.show()


# In[13]:


with open("data/UCF101_images/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01_0.jpg") as f:
    f.re


# In[ ]:




