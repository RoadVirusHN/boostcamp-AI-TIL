class Solution:
    def maximalSquare(self, matrix) -> int:
        
        mati, matj = len(matrix), len(matrix[0])

        for i in range(mati):
            matrix[i] = list(map(int, matrix[i]))

        dp = [[0 for _ in range(matj)] for __ in range(mati)]

        maxVal = 0
        for i in range(mati):
            dp[i][0] = matrix[i][0]
            if dp[i][0] == 1:
                maxVal = 1
        for j in range(matj):
            dp[0][j] = matrix[0][j]
            if dp[0][j] == 1:
                maxVal = 1
        for i in range(1, mati):
            for j in range(1, matj):          
                if matrix[i][j] == 1:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])+1
                    maxVal = maxVal if maxVal > dp[i][j] else dp[i][j] 
        return maxVal**2

a = Solution()
# print(a.maximalSquare([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))
# print(a.maximalSquare([["0","1"],["1","0"]]))
# print(a.maximalSquare([["0"]]))
print(a.maximalSquare([["1","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"],["1","1","1","1","1"],["0","0","1","1","1"]]))