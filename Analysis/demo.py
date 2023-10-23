import json

json_path = 'class_indices.json'
with open(json_path, "r") as f:
    class_indict = json.load(f)
with open('class_indices_TRUENAME.json', "r") as f:
    class_Name = json.load(f)
classes = []
for k, i in enumerate(class_indict):
    classes.append(class_Name[i])
print(classes)

