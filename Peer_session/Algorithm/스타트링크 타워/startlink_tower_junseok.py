import sys
sys.setrecursionlimit(987654321)
input = sys.stdin.readline

nums = '''###...#.###.###.#.#.###.###.###.###.###
#.#...#...#...#.#.#.#...#.....#.#.#.#.#
#.#...#.###.###.###.###.###...#.###.###
#.#...#.#.....#...#...#.#.#...#.#.#...#
###...#.###.###...#.###.###...#.###.###'''.split('\n')

nums = [[nums[j][i*4:i*4+3] for j in range(5)] for i in range(10)]
noSharps = [[[] for __ in range(3)] for _ in range(5)]
for i in range(5):
    for j in range(3):
        for numidx in range(10):
            if nums[numidx][i][j] == '.':
                noSharps[i][j].append(numidx)


numCount = int(input())
display = [input().rstrip('\n') for _ in range(5)]
display = [[display[j][i*4:i*4+3] for j in range(5)] for i in range(numCount)]

cases = [[] for _ in range(numCount)]

for idx in range(numCount):
    available =[i for i in range(10)]
    for i in range(5):
        for j in range(3):
            if display[idx][i][j] == '#':
                for unavail in noSharps[i][j]:
                    if unavail in available : available.remove(unavail)
    cases[idx] = [i for i in available]

answer = 0
caseNum = 1
for i in range(numCount):
    caseNum *= len(cases[i])
if caseNum:
    multiply = 1
    for caseIdx in range(len(cases)-1, -1, -1):
        answer += sum(cases[caseIdx])*(caseNum/len(cases[caseIdx])) * multiply
        multiply *= 10    
    print(answer/caseNum)
else:
    print(-1)
