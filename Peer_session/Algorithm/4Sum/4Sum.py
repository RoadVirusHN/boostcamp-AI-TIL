from typing import List
from collections import defaultdict

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        NumCount = {num:nums.count(num) for num in nums}
        half1 = nums[:len(nums)//2]
        half2 = nums[len(nums)//2:]
        half1Sums = defaultdict(lambda: [])
        half2Sums = defaultdict(lambda: [])
        half1Sums3 = defaultdict(lambda: [])
        half2Sums3 = defaultdict(lambda: [])
        answer = []
        half1NumCount = {num:half1.count(num) for num in half1}
        half2NumCount = {num:half2.count(num) for num in half2}
        
        for i in range(len(half1)):
            for j in range(i+1,len(half1)):
                half1Sums[half1[i]+half1[j]].append([half1[i], half1[j]])
                for k in range(j+1, len(half1)):
                    half1Sums3[half1[i]+half1[j]+half1[k]].append([half1[i], half1[j], half1[k]])
        for i in range(len(half2)):
            for j in range(i+1,len(half2)):
                half2Sums[half2[i]+half2[j]].append([half2[i], half2[j]])
                for k in range(j+1, len(half2)):
                    half2Sums3[half2[i]+half2[j]+half2[k]].append([half2[i], half2[j], half2[k]])

        for key, value in half1Sums.items():
            for arr1 in value:
                for arr2 in half1Sums.get(target-key) or []:
                    newarr = sorted(arr1+arr2)
                    newarrCount = {num:newarr.count(num) for num in newarr}
                    if newarr not in answer:
                        for num, count in newarrCount.items():
                            if count > half1NumCount[num]:
                                break
                        else:  
                            answer.append(newarr)

                for arr2 in half2Sums.get(target-key) or []:
                    newarr = sorted(arr1+arr2)
                    if newarr not in answer:
                        answer.append(newarr)

        for key, value in half2Sums.items():
            for arr1 in value:
                for arr2 in half1Sums.get(target-key) or []:
                    newarr = sorted(arr1+arr2)
                    if newarr not in answer:
                        answer.append(newarr)

                for arr2 in half2Sums.get(target-key) or []:
                    newarr = sorted(arr1+arr2)
                    newarrCount = {num:newarr.count(num) for num in newarr}
                    if newarr not in answer:
                        for num, count in newarrCount.items():
                            if count > half2NumCount[num]:
                                break
                        else:  
                            answer.append(newarr)
        
        for key, value in half1Sums3.items():
            for arr1 in value:
                if NumCount.get(target-key):
                    newarr = sorted(arr1+[target-key])
                    newarrCount = {num:newarr.count(num) for num in newarr}
                    if newarrCount[target-key] <= NumCount[target-key] and newarr not in answer:
                        answer.append(newarr)
        for key, value in half2Sums3.items():
            for arr1 in value:
                if NumCount.get(target-key):
                    newarr = sorted(arr1+[target-key])
                    newarrCount = {num:newarr.count(num) for num in newarr}
                    if newarrCount[target-key] <= NumCount[target-key] and newarr not in answer:
                        answer.append(newarr)

        return sorted(answer)

a = Solution()
# print(a.fourSum([1,0,-1,0,-2,2], 0))
# print(a.fourSum([], 0))
# print(a.fourSum([0,0,0,0], 1))
# print(a.fourSum([-3,-1,0,2,4,5], 0))
print(a.fourSum([0,-5,5,1,1,2,-5,5,-3], -11))