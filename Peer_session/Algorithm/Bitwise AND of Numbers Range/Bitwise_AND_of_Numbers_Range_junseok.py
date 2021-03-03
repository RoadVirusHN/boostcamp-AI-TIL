class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        n = format(n,f'#0b')[2:]
        maxLen = len(n)
        m = format(m,f'#0{maxLen+2}b')[2:]
        idx, answer =0, ''
        while idx<maxLen:
            if m[idx]==n[idx]: answer+=m[idx]
            else: break
            idx+=1
        answer = answer.ljust(maxLen,'0')
        return int(answer,2)






a = Solution()

print(a.rangeBitwiseAnd(5,7))
print(a.rangeBitwiseAnd(0,1))
print(a.rangeBitwiseAnd(0,0))
print(a.rangeBitwiseAnd(1,1))
print(a.rangeBitwiseAnd(1,2))
print(a.rangeBitwiseAnd(90,99))
print(a.rangeBitwiseAnd(5,5))