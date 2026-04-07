import sys
try:
    with open("inference_output.txt", encoding="utf-16-le") as f:
        content = f.read()
except:
    with open("inference_output.txt", encoding="utf-8") as f:
        content = f.read()
for line in content.splitlines():
    print(line)
