import bisect 

def upper_bound(arr, score):
    if score=='_':
        return len(arr)
    else:
        return len(arr[bisect.bisect_right(arr,score-1):])

def solution(infos, querys):
    scores = [[] for _ in range(1<<5)]
    appdict= {
        "chicken": 0,
        "junior": 1,
        "backend": 2,
        "cpp" : 3,
        "java": 4,
    }
    apps = ["chicken", "junior", "backend", "cpp","java"]
    for info in infos:
        lang, job, car, food, score = info.split()
        score = int(score)
        info_list = [lang, job, car, food]        
        result = 0
        for app in info_list:
            if app in apps:
                result = result ^ (1 << appdict[app])
        scores[result].append(score)    
    for score in scores:
        score.sort()

    answer = []
    query_dict = {}
    for query in querys:
        query = query.replace('and', '')
        lang, job, car, food, score = query.split()
        score = int(score)
        query_list = [food, car, job, lang]
        all = 0
        if query_dict.get(''.join(query_list)) == None:
            results = [0]
            for appidx in range(len(query_list)):
                if query_list[appidx] == "-":
                    newresults = [i for i in results]
                    for resultidx in range(len(results)):
                        results[resultidx] = results[resultidx] ^ (1 << appidx)
                    newresults += results
                    results = newresults
                    if appidx == 3:
                        appidx += 1
                        newresults = [i for i in results]
                        for resultidx in range(len(results)):
                            results[resultidx] = results[resultidx] ^ (1 << appidx)
                        newresults += results
                        results = newresults    
                else:
                    if query_list[appidx] in apps:
                        for idx in range(len(results)):
                            results[idx] = results[idx] ^ (1 << appdict[query_list[appidx]])   
            query_dict[''.join(query_list)] = results
        else:
            results = query_dict[''.join(query_list)]
        for result in results:
            all += upper_bound(scores[result],score)
        answer.append(all)
    return answer

infos = ["java backend junior pizza 150","python frontend senior chicken 210","python frontend senior chicken 150","cpp backend senior pizza 260","java backend junior chicken 80","python backend senior chicken 50"]
querys = ["java and backend and junior and pizza 100","python and frontend and senior and chicken 200","cpp and - and senior and pizza 250","- and backend and senior and - 150","- and - and - and chicken 100","- and - and - and - 150"]

# solution(infos, querys)
print(solution(infos, querys))
