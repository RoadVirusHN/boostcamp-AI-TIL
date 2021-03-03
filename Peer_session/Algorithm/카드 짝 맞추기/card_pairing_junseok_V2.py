# import heapq
INF = 987654321
answer = INF
def solution(board, r, c):
    global answer
    answer = INF
    fr = (r,c)
    recursive(fr, board, 0)
    return answer

def recursive(fr, board, dist):
    global answer
    isFinsish = True
    for i in range(4):
        for j in range(4):
            if board[i][j] != 0:
                isFinsish = False
                num = board[i][j]
                pairIdx = findVal((i,j), board)
                newfr, newto = (i,j), pairIdx
                newdist = dist+getDistance(board,fr,newfr)+getDistance(board,newfr,newto)+2
                if newdist > answer:
                    continue
                board[i][j], board[pairIdx[0]][pairIdx[1]] = 0, 0
                recursive(pairIdx, board, newdist)                 
                board[i][j], board[pairIdx[0]][pairIdx[1]] = num, num
    if isFinsish:
        answer = dist if dist < answer else answer

def findVal(fr, board):
    for i in range(4):
        for j in range(4):
            if (i,j)!=fr and board[i][j] == board[fr[0]][fr[1]]:
                return (i,j)


def getCtrlNearest(i,j, dir, board):
    while True:
        i += dir[0]
        j += dir[1]
        if not (i>=0 and i <4 and j >= 0 and j < 4):
            i -= dir[0]
            j -= dir[1]
            break
        elif not board[i][j] == 0:
            break
    return i, j


def updateNeighbor(i,j, board):
    delta = [(-1,0), (0, 1), (1,0), (0, -1)]
    neighbors = [(i+delta[k][0], j+delta[k][1]) for k in range(4) if i+delta[k][0] >= 0 and i + delta[k][0] < 4 and j+delta[k][1] >= 0 and j+delta[k][1] < 4]
    neighbors += [getCtrlNearest(i,j,delta[k],board) for k in range(4)]
    return neighbors


def getNeighbors(board):
    neighbors = [[[] for __ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            neighbors[i][j] = updateNeighbor(i,j,board)
    return neighbors


def getDistance(board,fr,to):
    if fr == to:
        return 0
    distance = [[INF for __ in range(4)] for _ in range(4)]
    neighbors = getNeighbors(board)
    visited = [[False for __ in range(4)] for _ in range(4)]
    distance[fr[0]][fr[1]] = 0
    visited[fr[0]][fr[1]] = True
    q = [(0,fr)]
    while q:
        nowDist, nowIdx = q.pop(0)
        for neighbor in neighbors[nowIdx[0]][nowIdx[1]]:
            if not visited[neighbor[0]][neighbor[1]]:
                distance[neighbor[0]][neighbor[1]] = nowDist+1
                if neighbor == to:
                    return distance[to[0]][to[1]]
                q.append((nowDist+1, neighbor))  
                visited[neighbor[0]][neighbor[1]] = True
    





print(solution([[1,0,0,0],[2,0,0,0],[0,0,0,2],[0,0,1,0]], 1, 0))
print(solution([[1,0,0,3],[2,0,0,0],[0,0,0,2],[3,0,1,0]], 1, 0))
print(solution([[3,0,0,2],[0,0,1,0],[0,1,0,0],[2,0,0,3]], 0, 1))
print(solution([[3,0,0,2],[4,0,1,0],[0,1,0,0],[2,4,0,3]], 0, 1))
print(solution([[3,4,4,2],[5,0,1,0],[0,1,0,0],[2,5,0,3]], 0, 1))
print(solution([[3,4,4,2],[5,0,1,6],[0,1,6,0],[2,5,0,3]], 0, 1))