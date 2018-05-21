def reverse():
    """
    reverse integer
    """
    x=-123
    y = abs(x)
    y = str(y)
    y = list(y)
    y = y[::-1]
    sign =cmp(x,0)       # = 1 if abs(x) == x else -1
    y = ''.join(y)
    y = int(y)
    if abs(y)<0xf7777777:    # y <= 2 ** 31 and y > -2 ** 31:
        return sign * y
    else:
        return 0

def add_two():
    n1 = 123
    n2 = 456
    list = []
    length = len(str(n1))
    n1 = 10 ** (length) + n1
    n2 = 10 ** (length) + n2
    n1_r = []
    n2_r = []
    for i in range(length):
        n1_r.append(n1 % 10)
        n2_r.append(n2 % 10)
        n1 //= 10
        n2 //= 10
    res=[0]
    for i in range(length):
        res.append(n1_r[i] + n2_r[i] + res[i])
        if res[i] >= 10:
            res[i + 1] = 1
            res[i] -= 10
    res.remove(res[0])
    return res

def roman():
    s = "MCMXCVI"
    num = []
    sign = []
    s = list(s)
    for i in range(len(s)):
        if s[i] == 'I':
            num.append(1)

        if s[i] == 'V':
            num.append(5)

        if s[i] == 'X':
            num.append(10)

        if s[i] == 'L':
            num.append(50)

        if s[i] == 'C':
            num.append(100)

        if s[i] == 'D':
            num.append(500)

        if s[i] == 'M':
            num.append(1000)
    num.append(0)
    sum = 0
    i = 0
    while i < len(num)-1:
        print(num[i],num[i+1])
        if num[i]>=num[i+1]:
            sum += num[i]
            i +=1
        elif num[i]<num[i+1]:
            sum = sum + num[i+1] - num [i]
            i +=2
    return sum

def longcompref(strs):
    lcp = ""
    if len(strs) == 0:
        return ""
    elif len(strs) == 1:
        return strs[0]
    else:
        for i in strs[0]:
            temp = strs[1:]
            for j in temp:
                if lcp + i not in j[:len(lcp + i)]:
                    return lcp
            lcp = lcp + i
        return lcp

def parentheses(s):
    s = list(s)
    map = {"]":"[", "}":"{",  ")":"("}
    # if s[0] in map.values():
    #     return False
    # temp = s[:]
    # i = 1
    # for i in s:
    #     if i in map.values():
    #         idx = temp.index(i)
    #         if len(temp) == 1 or map[temp[idx - 1]] != temp[idx]:
    #             return False
    #         else:
    #             del temp[idx]
    #             del temp[idx - 1]
    # if len(temp) == 0:
    #     return True
    # else:
    #     return False
    """
    fastest solution:
    """
    stack=[]
    for i in s:
        if i in map:
            if not stack or stack.pop() != map[i]:
                return False
        else:
            stack.append(i)
    return not stack

def removeDuplicates(nums):

    i = 0
    while i < len(nums):
        a = nums[i]
        j = len(nums)-1
        l = len(nums)
        while j >= i+1 and j < len(nums):
            if nums[j] == a:
                del nums[j]
                j = j - 1
        i = i + 1
    return nums

def hay_needle(haystack,needle):
    # fastest solution:
    # return haystack.find(needle)
    # or
    # if needle not in haystack:
    #     return -1
    # return haystack.index(needle)

    if needle == "":
        return 0
    for i in range(len(haystack)):
        j = 0
        while j < len(needle) and len(needle) <= len(haystack) - i:
            if needle[j] == haystack[i + j]:
                if j + 1 == len(needle):
                    return i
                j = j + 1
            else:
                break
    return -1

def countAndSay():
    n = 4
    oldset = "1"
    i = 1
    while i < n and n > 1:
        newset = ""
        # if i == 1:
        #     newset = str(len(oldset)) + oldset
        count = 1
        for j in range(len(oldset)):
            if j == len(oldset)-1:
                newset = newset + str(count) + oldset[j]
                break
            if oldset[j] == oldset[j + 1]:
                count = count + 1
            else:
                newset = newset + str(count) + oldset[j]
                count = 1
        i += 1
        oldset = newset
    return oldset

def comp(a, b, c):
    temp = [a, b, c]
    index = temp.index(max(temp))
    result = max(temp)
    return result, index

def maxsubarray(nums):
    longlist = nums
    longest = longlist
    longrcd = sum(longlist)
    while len(longlist) != 1:
        left = longlist[:-1]
        right = longlist[1:]
        if longlist[0] > longlist[-1]:
            longrcd = comp(left, longlist)[0]
            longest = comp(left, longlist)[1]
            longlist = longest
        elif longlist[0] < longlist[-1]:
            longrcd = comp(right, longlist)[0]
            longest = comp(right, longlist)[1]
            longlist = longest
        else:
            if left[0] > left[-1]:
                temprcd = comp(left[:-1], left)[0]
                templong = comp(left[:-1], left)[1]
            elif left[0] < left[-1]:
                temprcd = comp(left[1:], left)[0]
                templong = comp(left[1:], left)[1]
            if right[0] > right[-1]:
                temprcd_r = comp(left[:-1], left)[0]
                templong_r = comp(left[:-1], left)[1]
            elif right[0] < right[-1]:
                temprcd_r = comp(left[1:], left)[0]
                templong_r = comp(left[1:], left)[1]
    return 1

def length_of_last(s):
    count = 0
    for i in reversed(range(len(s))):
        if s(i) != " ":
            count = count + 1
        if s(i) == " " and count != 0:
            break
    return count

def plusOne(digits):
    """
    digits : [int] list
    faster solution

    c = ""
    for k in digits:
        c = c + str(k)
    f = int(c)
    f = f+1
    s = str(f)
    l = []
    for j in s:
        l.append(int(j))
    return l
    """

    l = len(digits)
    num = 0
    for i in range(l):
        num = num + digits[i] * (10 ** (l - i-1))
    num = num + 1
    res = []
    j = 0
    while num > 9:
        res.append(num % 10 * (10 ** j))
        num = num // 10
    res.append(num)
    res = res[::-1]
    return res

def addBinary(a, b):
    """

    num = int(a,2) + int(b,2)
        return bin(num)[2:]
    """
    ia = int(a)
    ib = int(b)
    sum = ia + ib
    if sum == 0:
        return str(sum)
    s = str(sum)
    temp = []
    for i in range(len(s)):
        temp.append(int(s[i]))
    sign = 0
    temp.insert(0,0)
    for i in range(len(temp)-1,-1,-1):
        if temp[i] > 1:
            temp[i] =temp[i]-2
            temp[i-1] += 1
    if temp[0] == 0:
        del temp[0]
    strc = "".join(str(x) for x in temp)
    return strc

def mySqrt(x):
    # b = float(x / 2)
    # if x == 1:
    #     return 1
    # while abs(b ** 2 - x) > 0.0001:
    #     if b ** 2 - x < 0:
    #         b = b + b / 2
    #     elif b ** 2 - x > 0:
    #         b = b / 2
    #     else:
    #         return int(b)
    # print(b ** 2 - x)
    # return int(b)
    """

    """
    # a = float(0)
    # b = float(x / 2)
    # if x == 1:
    #     return 1
    # while a < b:
    #     mid = a + (b - a) / 2
    #     if mid ** mid > x:
    #         b = mid - 1
    #     else:
    #         a = mid + 1
    # return int(b)

    # t = float(x / 2)
    t = float(x)/2
    print(t)
    while abs(t ** 2 - x) > 0.01:
        t = t - (t ** 2 - x) / 2 / t
    return int(t)

def climbStairs(n):
    f1=1
    f2=2
    if n ==1:
        return f1
    if n ==2:
        return f2
    i=3
    while i <=n:
        f = f1+f2
        f1 = f2
        f2 = f
        i+=1
    return f

def merge(nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: void Do not return anything, modify nums1 in-place instead.
    """
    for i in range(n):
        nums1[i+m] = nums2[i]
    nums1.sort()

    def func(nums, l, r):
        if l > r:
            return
        else:
            node = TreeNode(nums[(l + r) / 2])
            node.left = func(nums, l, (l + r) / 2 - 1)
            node.right = func(nums, (l + r) / 2 + 1, r)
            return node

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def func(nums, l, r):
    if l > r:
        return
    else:
        node = TreeNode(nums[(l + r) / 2])
        node.left = func(nums, l, (l + r) / 2 - 1)
        node.right = func(nums, (l + r) / 2 + 1, r)
        return node

class Solution(object):

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return func(nums, 0, len(nums) - 1)

if __name__=="__main__":
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    print(root.val)
