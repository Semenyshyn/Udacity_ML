import re

a = '232 Toy4444 Story (1995) 345'

b = re.findall("\((\d{4})\)", a)[0]
print(b)
