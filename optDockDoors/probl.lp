\ENCODING=ISO-8859-1
\Problem name: MIP1

Minimize
 obj1: x0_0 + x0_1 + x0_2 + x0_3 + x1_0 + x1_1 + x1_2 + x1_3
Subject To
 a0:   x0_0 + x1_0  = 1
 a1:   x0_1 + x1_1  = 1
 a2:   x0_2 + x1_2  = 1
 a3:   x0_3 + x1_3  = 1
 cap0: x0_0 + x0_1 + x0_2 + x0_3 <= 12
 cap1: x1_0 + x1_1 + x1_2 + x1_3 <= 9
Bounds
 0 <= x0_0 <= 1
 0 <= x0_1 <= 1
 0 <= x0_2 <= 1
 0 <= x0_3 <= 1
 0 <= x1_0 <= 1
 0 <= x1_1 <= 1
 0 <= x1_2 <= 1
 0 <= x1_3 <= 1
End
