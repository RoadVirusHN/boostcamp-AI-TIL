import sys
input = sys.stdin.readline
sys.setrecursionlimit(987654321)
def dfs(i):
    visited[i] = True
    for node in edges[i]:
        if not visited[node]:
            dfs(node)
    st.append(i)

def getSCC(node):
    visited[node] = True
    SCC = [node]
    for adj in revedges[node]:
        if not visited[adj]:
            SCC += getSCC(adj)
    return SCC

caseCount = int(input())
for case in range(caseCount):
    nodeCount, edgeCount = map(int, input().split())
    edges = [[] for i in range(nodeCount+1)]
    revedges = [[] for i in range(nodeCount+1)]
    for _ in range(edgeCount):
        fr, to = map(int, input().split())
        edges[fr].append(to)
        revedges[to].append(fr)
    visited = [False for _ in range(nodeCount+1)]
    st = []
    for i in range(1, nodeCount+1):
        if not visited[i]:
            dfs(i) 
    visited = [False for _ in range(nodeCount+1)]
    answer = 0
    while st:
        now = st.pop()
        if not visited[now]:
            SCC = getSCC(now)
            startSCC = True
            for comp in SCC:
                for i in revedges[comp]:
                    if i not in SCC:
                        startSCC = False
                        break
                if not startSCC:
                    break
            if startSCC:
                answer += 1
    print(answer)