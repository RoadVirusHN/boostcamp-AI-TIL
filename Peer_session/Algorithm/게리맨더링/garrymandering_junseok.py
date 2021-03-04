import sys
import itertools
input = sys.stdin.readline
areaNum = int(input())
pops = list(map(int, input().split()))
graph = [list(map(lambda x: x-1,list(map(int, input().split()))[1:])) for _ in range(areaNum)]


print(list(list(itertools.combinations([i for i in range(areaNum)],j)) for j in range(areaNum//2, 0, -1)))
