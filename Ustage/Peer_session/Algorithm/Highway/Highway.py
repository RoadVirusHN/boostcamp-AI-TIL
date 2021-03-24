import sys, functools
input = sys.stdin.readline

def ccw(a, b, c):
    t = (b[0]-a[0])*(c[1]-a[1]) - (c[0] - a[0])*(b[1] - a[1])
    if t > 0:
        return 1
    elif t < 0:
        return -1
    else :
        return 0


def comp(a, b):
    return -ccw(fp,a,b)

caseCount = int(input())
for case in range(caseCount):
    cityCount = int(input())
    cities = [[] for _ in range(cityCount)]
    for city in range(cityCount):
         cities[city] = list(map(int, input().split()))
    print(cities)
    cities.sort(key=lambda x : x[0])
    fp = cities[0]
    cities.sort(key=functools.cmp_to_key(comp))
    print(cities)
    st = [cities[0],cities[1]]
    for city in cities[2:]:
        while(len(st) >= 2):
            if (ccw(st[-2], st[-1], city) > 0):
                break
            st.pop()
        st.append(city)
    maxlen = 0
    maxPair = []
    for idx1 in range(len(st)):
        for idx2 in range(idx1, len(st)):
            for city in st:
                if city != st[idx1] and city != st[idx2]:
                    if ccw(st[idx1], st[idx2], city):

