import tokenize
import re

statement = "$main[-1].step%100==0".replace(" ", "")
print(statement)
print('_'.isidentifier())


print(parsing(statement))