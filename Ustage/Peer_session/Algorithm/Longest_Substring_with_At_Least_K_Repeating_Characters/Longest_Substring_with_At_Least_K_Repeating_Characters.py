import sys, re
sys.setrecursionlimit(987654321)

class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        self.answer = 0
        self.recur(s,k)
        return self.answer
    
    def recur(self, s, k):
        if len(s) <= self.answer:
            return
        strCount = {char:s.count(char) for char in set(s)}        
        nono = [key for key, val in strCount.items() if val < k]
        if nono:
            s = re.split(str(nono),s)
            for sub in s:
                self.recur(sub,k)
        else:
            result = len(s)
            self.answer = result if result > self.answer else self.answer

a = Solution()
print(a.longestSubstring("aaabb",3))
print(a.longestSubstring("ababbc",2))