 import os

rootdir ='C:/Users/sid/Desktop/test'
main_dir = '/Users/AR_Magnavox/Documents/RUDSSP FILES/fake'        # Main dir
save_dir = os.path.join(main_dir,'signalmedia_data' )
dict1 = {}

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        f=open(file,'r')
        lines=f.readlines()
        f.close()
        f=open(file,'w')
        for line in lines:
            newline = "No you are not"
            f.write(newline)
        f.close()