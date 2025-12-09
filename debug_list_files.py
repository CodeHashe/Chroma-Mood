import os

print(f"CWD: {os.getcwd()}")
print("Files in CWD:")
for f in os.listdir('.'):
    print(f)
