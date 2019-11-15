import glob

dir_lists = glob.glob("*")
prodir = []

for dir in dir_lists:
  file_lists = glob.glob(dir+"/*")
  if len(file_lists) == 100:
    print(dir," ok")
  else:
    print(dir,len(file_lists))
    prodir.append(dir)

print(prodir)  
