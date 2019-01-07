import os

file_path="E:\\02 撰写论文"

file_list=os.listdir(file_path)
print(file_list)
i=1
for file in file_list:
    if i<10:
        newfile='0'+str(i)+' '+file

    else:
        newfile=str(i)+' '+file
    i=i+1
    os.rename(os.path.join(file_path,file),os.path.join(file_path,newfile))