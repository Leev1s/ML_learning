from email import header
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   d1.py
@Time    :   2022/08/08 08:39:37
@Author  :   Lev1s
@Version :   1.0
@Contact :   Lev1sStudio.cn@gmail.com
@PW      :   http://Lev1s.cn
@Github  :   https://github.com/o0Lev1s0o

'''
print('''
    __             ___        _____ __            ___     
   / /   ___ _   _<  /____   / ___// /___  ______/ (_)___ 
  / /   / _ \ | / / / ___/   \__ \/ __/ / / / __  / / __ \\
 / /___/  __/ |/ / (__  )   ___/ / /_/ /_/ / /_/ / / /_/ /
/_____/\___/|___/_/____/   /____/\__/\__,_/\__,_/_/\____/
''')
# here put the import lib
import torch as t
import numpy as np

x = t.rand(5,8)
print(x)

y = t.arange(0,100,1)
print(y)

z = t.eye(10)
print(z)

z = z[:,1:2]
print(z)