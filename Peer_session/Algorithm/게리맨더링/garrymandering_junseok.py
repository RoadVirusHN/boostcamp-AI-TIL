import sys
import itertools

def isSCC(vertices):
    global graph
    visited = [False for _ in range(len(vertices))]
    q = [vertices[0]]
    visited[0] = True
    while q:
        now = q.pop(0)
        for neighbor in graph[now]:
            if neighbor in vertices and not visited[vertices.index(neighbor)]:
                visited[vertices.index(neighbor)] = True
                q.append(neighbor)
    if False in visited:
        return False
    else:
        return True

input = sys.stdin.readline
areaNum = int(input())
pops = list(map(int, input().split()))
allPops = sum(pops)
minimum = allPops + 1
graph = [list(map(lambda x: x-1,list(map(int, input().split()))[1:])) for _ in range(areaNum)]
areas = list(itertools.chain.from_iterable((itertools.combinations([i for i in range(areaNum)],j)) for j in range(areaNum//2, 0, -1)))
# itertools.chain.from_iterable(): squeeze
visited = []
for group in areas:
    group = list(group)
    groupPop = sum([pops[popidx] for popidx in group])
    if abs(allPops - 2*groupPop) < minimum:
        if isSCC(group) and isSCC([i for i in range(areaNum) if i not in group]):
            minimum = abs(allPops - 2*groupPop)
if minimum == allPops + 1:
    print(-1)
else:
    print(minimum)

while True:
    a = input()
    print(a)




