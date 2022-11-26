f = open("Dataset_Topics.txt", "r")

# We create a dictionary where the key is a school subject
# and the value is the list of the words related to this subject
dataset = {"biology": list(dict.fromkeys(f.readline().split(";"))),
           "compsci": list(dict.fromkeys(f.readline().split(";"))),
           "physics": list(dict.fromkeys(f.readline().split(";"))),
           "chemistry": list(dict.fromkeys(f.readline().split(";"))),
           "philosophy": list(dict.fromkeys(f.readline().split(";")))}
f.close()


