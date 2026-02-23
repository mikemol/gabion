# Exact rational harness placeholder.
# See conversation-derived scripts for full implementation; this file is a starting point.

from fractions import Fraction

def v_p(p:int, n:int) -> int:
    e=0
    while n%p==0 and n>0:
        n//=p; e+=1
    return e

if __name__ == "__main__":
    print("Stub: fill with exact rational atlas harness")
