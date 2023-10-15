# 力扣Hot100

## 1.两数之和 Easy

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *target* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**示例 1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**示例 2：**

```
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**示例 3：**

```
输入：nums = [3,3], target = 6
输出：[0,1]
```

**提示：**

- `2 <= nums.length <= 104`
- `-109 <= nums[i] <= 109`
- `-109 <= target <= 109`
- **只会存在一个有效答案**

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        //讨论特殊情况
        if (nums == null || nums.length == 0) {
            return res;
        }
        //使用一个Map来记录数组的值和索引值
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            //定义一个临时变量
            int temp = target  - nums[i];
            //更新res数组
            if (map.containsKey(temp)){
                res[0] = i;
                res[1] = map.get(temp);
                break;//结束循环
            }
            map.put(nums[i],i);
        }
        return res;
    }
}
```

## 2.两数相加 Medium

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例 1：**

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

**示例 2：**

```
输入：l1 = [0], l2 = [0]
输出：[0]
```

**示例 3：**

```
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

**提示：**

- 每个链表中的节点数在范围 `[1, 100]` 内
- `0 <= Node.val <= 9`
- 题目数据保证列表表示的数字不含前导零

Related Topics

- 递归
- 链表
- 数学

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        //辅助的哑节点
        ListNode pre = new ListNode(-1);
        //辅助遍历的节点
        ListNode cur = pre;
        //carry表示十位，要么是1要么是0，这里的十位是指两个指针指向的两个数相加后的十位
        int carry = 0;
        while (l1 != null || l2 != null){
            //获取节点值，如果节点为空，就是0
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y +carry;
            //计算十位
            carry = sum / 10;
            //计算个位
            sum = sum % 10;
            //添加节点
            cur.next = new ListNode(sum);
            //节点往后移
            cur = cur.next;
            if (l1 != null){
                l1 = l1.next;
            }
            if (l2 != null){
                l2 = l2.next;
            }
        }
        //如果到最后十位还是1的话，得补一个节点
        if (carry == 1){
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }
}
```

## 3.无重复字符的最长子串 Medium

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1:**

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

**提示：**

- `0 <= s.length <= 5 * 104`
- `s` 由英文字母、数字、符号和空格组成

Related Topics

- 哈希表

- 字符串

- 滑动窗口


```java
class Solution {
    public static int lengthOfLongestSubstring(String s) {
        //特殊情况，字符串长度为0
        if (s.length() == 0) return 0;
        //使用map记录当前滑动窗口是否存在当前遍历到的字符
        Map<Character,Integer> map = new HashMap<>();
        //转为数组
        char[] chars = s.toCharArray();
        int res = 0;
        //滑动窗口左边界
        int left = 0;
        //右边界
        int right = 0;
        while (right < chars.length){
            //不包含当前字符，我们就记录一下，右边界右移
            if (!map.containsKey(chars[right]) ){
                map.put(chars[right],1);
                right++;
            }else {
                //说明包含了，map先删除这个字符
                //左边界右移
                map.remove(chars[left]);
                left++;
            }
            //记录滑动窗口的最大值
            //因为前面right已经++了，所以这里right-left不用+1
            res = Math.max(res,right-left);
        }
        return res;
    }
}

```

```java
//对上一个方法进行优化，上一个方法里是left的更新比较慢，因为一旦遇到重复之后就需要更新left到重复字符的下一个位置，所以在这里对这个细节进行优化
//举个例子，假设字符串是 "abcba"：当遍历到第二个 "b" 时，发现它在位置 3 和位置 1 都出现过，此时需要更新 start 的值。通过 map.get(c) 可以得到最近一次 "b" 出现的位置为 1，那么通过 Math.max(map.get(c), start) 就可以得到新的 start 值为 1+1 = 2，即重复字符的下一个位置。
class Solution {
    public static int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) return 0;
        Map<Character,Integer> map = new HashMap<>();
        //左边界
        int start = 0;
        int res = 0;
        //end右边界
        for (int end = 0; end < s.length();end++){
            char c = s.charAt(end);
            if (map.containsKey(c)){
                //左边界直接跳到右边界
                start = Math.max(map.get(c),start);
            }
            res = Math.max(res,end-start+1);
            //end+1是表示记录当前字符索引的下一位，这样就能直接跳到重复字符的下一位了
            map.put(s.charAt(end),end+1);
        }
        return res;
    }
}
```

## 4.寻找两个正序数组的中位数 Hard

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

算法的时间复杂度应该为 `O(log (m+n))` 。

**示例 1：**

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

**示例 2：**

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

**提示：**

- `nums1.length == m`
- `nums2.length == n`
- `0 <= m <= 1000`
- `0 <= n <= 1000`
- `1 <= m + n <= 2000`
- `-106 <= nums1[i], nums2[i] <= 106`

Related Topics

- 数组
- 二分查找
- 分治

```java
//合并数组，但只要合并到中间位置就可以了
class Solution {
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int l = m + n;
        int i = 0, j = 0;
        double[] res = new double[l];
        int index = 0;
        while (i < m && j < n && i + j <= l / 2){
            if (nums1[i] <= nums2[j]){
                res[index++] = nums1[i++];
            }else {
                res[index++] = nums2[j++];
            }
        }
        //有可能只是一个数组元素很少，遍历完了，但是还没遍历到合并数组的中间位置
        while (i == m && i + j <= l /2){
            res[index++] = nums2[j++];
        }
        while (j == n && i + j <= l /2){
            res[index++] = nums1[i++];
        }
        if (l %2 == 0){
            return (res[l/2] + res[l/2 - 1]) / 2;
        }else {
            return res[l/2];
        }
    }
}
```

## 5.最长回文子串 Medium

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**示例 2：**

```
输入：s = "cbbd"
输出："bb"
```

**提示：**

- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母组成

Related Topics

- 字符串
- 动态规划

```java
class Solution {
    public String longestPalindrome(String s) {
        int len = s.length();
        //特殊情况:如果字符串长度只有1，那一定是回文子串
        if (len < 2) return s;
        //dp表示在s[i...j]范围内是否是回文子串
        boolean[][] dp = new boolean[len][len];
        //初始化：根据dp的含义，而且我们知道单个字符肯定是回文子串
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }

        char[] chars = s.toCharArray();

        int begin = 0;//记录回文子串的起始位置
        int maxLen = 1;//记录回文子串的最大长度

        //遍历
        //用动态规划dp[i][j]是否为true，是取决于dp[i+1][j-1]的
        //所以这里我们先遍历列，再遍历行
        for (int j = 1; j < len; j++) {
            for (int i = 0; i < j; i++) {
                //两个字符不同，那肯定不是回文
                if (chars[i] != chars[j]){
                    dp[i][j] = false;
                }else {
                    //dp[i][j]是否为true，是取决于dp[i+1][j-1]的
                    //判断一下边界条件，(j-1)-(i+1)+1 >= 2 才能用状态转移方程
                    //不然的话，如果长度小于2,要么为1要么为0，再加上当前i和j的字符是相同的
                    //那么dp[j][j]肯定是true了
                    if (j - i < 3){
                        dp[i][j] = true;
                    }else {
                        dp[i][j] = dp[i+1][j-1];
                    }
                }
                //如果当前是回文子串，并且长度也是大于maxLen
                //那么就更新begin 和 maxLen
                if (dp[i][j] && j - i + 1 > maxLen){
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }

        return s.substring(begin,begin+maxLen);
    }
}
```

## 6.正则表达式匹配 Hard

给你一个字符串 `s` 和一个字符规律 `p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。

- `'.'` 匹配任意单个字符
- `'*'` 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 **整个** 字符串 `s`的，而不是部分字符串。

**示例 1：**

```
输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入：s = "aa", p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**示例 3：**

```
输入：s = "ab", p = ".*"
输出：true
解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

**提示：**

- `1 <= s.length <= 20`
- `1 <= p.length <= 20`
- `s` 只包含从 `a-z` 的小写字母。
- `p` 只包含从 `a-z` 的小写字母，以及字符 `.` 和 `*`。
- 保证每次出现字符 `*` 时，前面都匹配到有效的字符

Related Topics

- 递归
- 字符串
- 动态规划



动态规划：https://www.bilibili.com/video/BV1jd4y1U7kE/?spm_id_from=333.337.search-card.all.click&vd_source=92cb5cb9faa01574e9b1f82bf91d080d

https://www.iamshuaidi.com/2032.html

```java
class Solution {
    public boolean isMatch(String s, String p) {
        if (s == null || p == null){
            return true;
        }
        int n = s.length();
        int m = p.length();
        boolean[][] dp = new boolean[n+1][m+1];
        dp[0][0] = true;
        for (int j = 1; j <= m ; j++) {
            if (p.charAt(j-1) == '*'){
                dp[0][j] = dp[0][j-2];
            }
        }
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                if(p.charAt(j-1)=='.'||p.charAt(j-1)==s.charAt(i-1))
                    dp[i][j] = dp[i-1][j-1];
                else if(p.charAt(j-1)=='*'){
                    if(j!=1&&p.charAt(j-2)!='.'&&p.charAt(j-2)!=s.charAt(i-1)){
                        dp[i][j] = dp[i][j-2];
                    }else{
                        dp[i][j] = dp[i][j-2] || dp[i][j-1]||dp[i-1][j];
                    }
                }
            }
        }
        return dp[n][m];
    }
}
```

## 7.盛最多水的容器 Medium

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

**示例 1：**

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**示例 2：**

```
输入：height = [1,1]
输出：1
```

**提示：**

- `n == height.length`
- `2 <= n <= 105`
- `0 <= height[i] <= 104`

Related Topics

- 贪心
- 数组
- 双指针

```java
class Solution {
    public int maxArea(int[] height) {
        //考虑特殊情况
        if (height.length < 2) return 0;
        //返回的结果
        int res = 0;
        //双指针
        int left = 0, right = height.length-1;
        while (left < right){
            //h是高
            int h = Math.min(height[left],height[right]);
            //计算面积
            int area = h * (right - left);
            //更新res
            res = Math.max(res,area);
            //如果左指针对应的高度比h低的话，指针右移
            while (left < right && height[left] <= h){
                left++;
            }
            //如果右指针对应的高度比h低的话，指针左移
            while (left < right && height[right] <= h){
                right--;
            }
        }
        return res;
    }
}
```

## 8.三数之和 Medium

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请

你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

**示例 1：**

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
```

**示例 2：**

```
输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
```

**示例 3：**

```
输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。
```

**提示：**

- `3 <= nums.length <= 3000`
- `-105 <= nums[i] <= 105`

Related Topics

- 数组
- 双指针
- 排序

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        //排序，从小到大
        Arrays.sort(nums);
        //数组长度
        int n = nums.length;
        //遍历第一个数，只需要到n-2，n-1是第三个数在遍历
        for (int i = 0; i < n - 2; i++) {
            //跳过重复数字
            if (i > 0 && nums[i] == nums[i-1]){
                continue;
            }
            //优化细节1：如果前三个数加起来都大于0了，那么就不可能存在三个数加起来能等于0
            if (nums[i] + nums[i+1] + nums[i+2] > 0) {
                return res;
            }
            //优化细节1，如果最小的和最大的两个加起来小于0的话，那就没必要去遍历第二个数和第三个数了，因为这两个数比最大的两个数小
            //直接进入下一循环
            if (nums[i] + nums[n-2] + nums[n-1] < 0){
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right){
                if (nums[i] + nums[left] + nums[right] == 0){
                    res.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    //去重第二个数字
                    while (left < right && nums[right] == nums[right-1]){
                        right--;
                    }
                    //去重第三个数
                    while (left < right && nums[left] == nums[left+1]){
                        left++;
                    }
                    //移动指针
                    left++;
                    right--;
                }else if (nums[i] + nums[left] + nums[right] < 0){
                    left++;
                }else {
                    right--;
                }
            }
        }
        return res;
    }
}
```

## 9.电话号码的字母组合 Medium

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/11/09/200px-telephone-keypad2svg.png)

**示例 1：**

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**示例 2：**

```
输入：digits = ""
输出：[]
```

**示例 3：**

```
输入：digits = "2"
输出：["a","b","c"]
```

**提示：**

- `0 <= digits.length <= 4`
- `digits[i]` 是范围 `['2', '9']` 的一个数字。

Related Topics

- 哈希表
- 字符串
- 回溯

```java
class Solution {
    List<String> res = new LinkedList<>();
    StringBuffer sb = new StringBuffer();
    String[] numsStr = new String[]{
            "",
            "",
            "abc",
            "def",
            "ghi",
            "jkl",
            "mno",
            "pqrs",
            "tuv",
            "wxyz"
    };
    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) {
            return res;
        }
        dfs(digits,0);
        return res;
    }

    public void dfs(String digits, int index){
        //已经递归到这个数字对应的最后一个字母了，那就记录
        if (index == digits.length()) {
            res.add(sb.toString());
            return;
        }
        //获取这个数字对应的字母字符串
        String numChar = numsStr[digits.charAt(index)-'0'];
        //遍历字母字符
        for (int i = 0; i < numChar.length(); i++) {
            //记录该字母
            sb.append(numChar.charAt(i));
            //递归
            dfs(digits,index+1);
            //回溯
            sb.deleteCharAt(sb.length()-1);
        }

    }

}
```

```java
//回溯的另一种写法
class Solution {
    public static final String[] MAPPING = new String[]{
            "",
            "",
            "abc",
            "def",
            "ghi",
            "jkl",
            "mno",
            "qprs",
            "tuv",
            "wxyz"
    };

    List<String> res = new LinkedList<>();
    char[] path, digits;

    public List<String> letterCombinations(String digits) {
        int n = digits.length();
        if (n == 0) return res;
        this.digits = digits.toCharArray();
        path = new char[n];
        dfs(0);
        return res;
    }

    public void dfs(int i){
        if (i == digits.length) {
            res.add(new String(path));
            return;
        }
        //比如这里的是digits是[2,3]
        //那么MAPPING[2]对应的就是"abc"
        for (char c : MAPPING[digits[i] -'0'].toCharArray()){
            //这里path中的元素虽然没有被删掉，但在下一层循环中会被覆盖
            path[i] = c;
            dfs(i+1);
        }
    }
}
```



## 10.删除链表倒数第N个节点 Medium

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**示例 1：**

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

**示例 2：**

```
输入：head = [1], n = 1
输出：[]
```

**示例 3：**

```
输入：head = [1,2], n = 1
输出：[1]
```

**提示：**

- 链表中结点的数目为 `sz`
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`

**进阶：**你能尝试使用一趟扫描实现吗？

Related Topics

- 链表
- 双指针

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        //哑结点的好处在于可以不用讨论头结点被删除的情况
        ListNode dummy = new ListNode(-1,head);
        ListNode temp = dummy;
        //获取链表长度
        int len = getLength(head);
        int k = len - n ;
        for (int i = 0; i <= k; i++) {
            if (i == k){
                temp.next = temp.next.next;
                break;
            }
            temp = temp.next;
        }
        return dummy.next;
    }

    public int getLength(ListNode head){
        int len = 0;
        while (head != null){
            len++;
            head = head.next;
        }
        return len;
    }
}
```

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1, head);
        ListNode left = dummy;
        ListNode right =dummy;
        //让right指针先走n步
        while (n-- > 0){
            right = right.next;
        }
        //然后left和right一起走，直到right到最后一个节点，这个时候left所指向的就是待删除节点的前一个节点
        while (right.next != null){
            left = left.next;
            right = right.next;
        }
        left.next = left.next.next;
        return dummy.next;
    }
}
```

## 11.有效的括号 Easy

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**示例 1：**

```
输入：s = "()"
输出：true
```

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```
输入：s = "(]"
输出：false
```

**提示：**

- `1 <= s.length <= 104`
- `s` 仅由括号 `'()[]{}'` 组成

Related Topics

- 栈
- 字符串

```java
class Solution {
    public boolean isValid(String s) {
        char[] chars = s.toCharArray();
        Deque<Character> deque = new LinkedList<>();
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            //遇到左括号，就把右括号放入队列中
            if (c == '(') {
                deque.push(')');
            }else if (c == '[') {
                deque.push(']');
            }else if (c == '{') {
                deque.push('}');
            }else if (deque.isEmpty() || deque.peek() != c){
                //因为当前已经遍历到字符并且这个字符不是左括号，但是队列已经为空了，说明右括号多了
                //如果头元素和当前字符不一样，要么是左括号多了，要么就是左右括号不能匹配
                return false;
            }else {//如果遇到队列头元素和当前字符相同，就把头元素删掉
                deque.pop();
            }
        }
        return deque.isEmpty();
    }
}
```

## 12.合并两个有序列表 Easy

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例 1：**

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**示例 2：**

```
输入：l1 = [], l2 = []
输出：[]
```

**示例 3：**

```
输入：l1 = [], l2 = [0]
输出：[0]
```

**提示：**

- 两个链表的节点数目范围是 `[0, 50]`
- `-100 <= Node.val <= 100`
- `l1` 和 `l2` 均按 **非递减顺序** 排列

Related Topics

- 递归
- 链表

```java
//最简单的办法是新造一个链表，遍历这两个链表，比较两个节点的大小，用小节点的值创建新的节点，连接在新链表之后
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (list1 != null && list2 != null){
            if (list1.val <= list2.val) {
                cur.next = new ListNode(list1.val);
                list1 = list1.next;
            } else {
                cur.next = new ListNode(list2.val);
                list2 = list2.next;
            }
            cur = cur.next;
        }
        if (list1 != null) {
            cur.next = list1;
        }
        if (list2 != null){
            cur.next = list2;
        }
        return dummy.next;
    }
}
```

```java
//方法二，使用原来链表的节点去造新链表
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (list1 != null && list2 != null){
            if (list1.val <= list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        if (list1 != null) {
            cur.next = list1;
        }
        if (list2 != null){
            cur.next = list2;
        }
        return dummy.next;
    }
}
```

```java
//使用递归
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        if (list1.val <= list2.val) {
            list1.next = mergeTwoLists(list1.next,list2);
            return list1;
        }else {
            list2.next = mergeTwoLists(list1,list2.next);
            return list2;
        }
    }
}
```

## 13.括号生成 Medium

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

**示例 1：**

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

**示例 2：**

```
输入：n = 1
输出：["()"]
```

**提示：**

- `1 <= n <= 8`

Related Topics

- 字符串
- 动态规划
- 回溯

```java
class Solution {
	List res = new LinkedList<>(); // 保存结果的集合
	char[] path; // 保存括号组合的数组
	int n; // 括号对数

	public List<String> generateParenthesis(int n) {
    path = new char[2 * n]; // 初始化括号组合数组的长度
    this.n = n; // 保存括号对数
    dfs(0,0); // 开始深度优先搜索，以0作为起始位置，初始括号数为0
    return res; // 返回结果集合
	}

	public void dfs(int i, int open) {
    	if (i == 2 *n){ // 当组合的括号对数达到2n时，即所有位置都填满了括号
        	res.add(new String(path)); // 将当前的括号组合转化为字符串，并添加进结果集合
        	return; // 结束当前的深度优先搜索
    	}

    	if (open < n) { // 如果当前开放的左括号数小于n
        	path[i] = '('; // 在当前位置添加一个左括号
        	dfs(i+1,open+1); // 继续下一个位置的搜索，并更新左括号数
    	}
    	if (i - open < open) { // 如果当前位置之前的右括号数小于左括号数
        	path[i] = ')'; // 在当前位置添加一个右括号
        	dfs(i+1,open); // 继续下一个位置的搜索，并保持左括号数不变
    	}
	}
}
```

## 14.合并K个升序链表 Hard

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例 1：**

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**

```
输入：lists = []
输出：[]
```

**示例 3：**

```
输入：lists = [[]]
输出：[]
```

**提示：**

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- `lists[i]` 按 **升序** 排列
- `lists[i].length` 的总和不超过 `10^4`

Related Topics

- 链表
- 分治
- 堆（优先队列）
- 归并排序

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        //优先队列，默认情况下，最小的数排在队列头部
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for (ListNode listNode : lists) {//将所有节点值塞进优先队列中，这样就得到了从小到大的排序
            while (listNode != null) {
                queue.add(listNode.val);
                listNode = listNode.next;
            }
        }

        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (!queue.isEmpty()) {//通过遍历队列，创建节点
            cur.next = new ListNode(queue.poll());
            cur = cur.next;
        }
        return dummy.next;

    }
}
```

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        if (lists.length == 1) {
            return lists[0];
        }
        return mergeKListNodes(lists, 0, lists.length - 1);
    }

    // 合并 lists 数组中区间 [i, j] 的链表
    public ListNode mergeKListNodes(ListNode[] lists, int i, int j) {
        if (i == j) {
            return lists[i]; // 当 i 和 j 相等时，直接返回该链表
        }
        int mid = i + (j - i) / 2;
        ListNode left = mergeKListNodes(lists, i, mid); // 递归合并左侧区间 [i, mid]
        ListNode right = mergeKListNodes(lists, mid + 1, j); // 递归合并右侧区间 [mid + 1, j]
        return mergeTwoListNode(left, right); // 合并左右两个链表
    }

    // 合并两个有序链表
    public ListNode mergeTwoListNode(ListNode listNode1, ListNode listNode2) {
        if (listNode1 == null) {
            return listNode2;
        }
        if (listNode2 == null) {
            return listNode1;
        }
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (listNode1 != null && listNode2 != null) {
            if (listNode1.val <= listNode2.val) {
                cur.next = listNode1; // 将当前节点连接到合并后的链表中
                listNode1 = listNode1.next; // 移动 listNode1 的指针
            } else {
                cur.next = listNode2; // 将当前节点连接到合并后的链表中
                listNode2 = listNode2.next; // 移动 listNode2 的指针
            }
            cur = cur.next; // 移动合并后的链表指针
        }
        cur.next = listNode1 == null ? listNode2 : listNode1; // 将剩余的节点连接到合并后的链表中
        return dummy.next; // 返回合并后的链表
    }
}
```

## 15.下一个排列 Medium

整数数组的一个 **排列** 就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须 **原地** 修改，只允许使用额外常数空间。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

**示例 3：**

```
输入：nums = [1,1,5]
输出：[1,5,1]
```

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 100`

Related Topics

- 数组
- 双指针

```java
public static void nextPermutation(int[] nums) {
    int len = nums.length;
    int left = len - 2; // 从倒数第二个元素开始向左遍历
    int right = len - 1; // 从倒数第一个元素开始向左比较
    while (left >= 0) { // 遍历数组中的每一个元素
        if (nums[left] < nums[right]) { // 找到两个相邻元素，并且左边的比右边的小
            for (int i = len - 1; i >= right; i--) { // 从右向左找到第一个大于较小元素的元素
                if (nums[i] > nums[left]) {
                    int temp = nums[i];
                    nums[i] = nums[left];
                    nums[left] = temp;
                    Arrays.sort(nums, right, len); // 对较小元素右侧的元素进行排序
                    break;
                }
            }
            break; // 执行了上面的代码就可以结束遍历了
        }
        left--;
        right--;
    }
    if (left < 0) { // 如果遍历到left<0,说明就找不到下一个排列，那就直接从小到大排序
        Arrays.sort(nums);
    }
}
```

```java
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        //寻找相邻的两个数，并且左边的数小于右边的数
        while (i >= 0 && nums[i] >= nums[i+1]){//只要有一个条件不满足，就停止循环
            i--;
        }
        //只有i大于等于0，才说明找到了那两个相邻的数
        if (i >= 0) {
            //j从后往前遍历，找第一个比nums[i]大的数
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            //交换这两个数
            swap(nums,i,j);
        }
        //将i往后的数进行从小到大排序
        reverse(nums,i+1);
    }

    public void swap(int[] nums,int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int index){
        int left = index;
        int right = nums.length - 1;
        while (left < right) {
            swap(nums,left,right);
            left++;
            right--;
        }
    }
}
```

## 16.最长有效括号 Hard

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

**提示：**

- `0 <= s.length <= 3 * 104`
- `s[i]` 为 `'('` 或 `')'`

Related Topics

- 栈
- 字符串
- 动态规划

官方题解：https://leetcode.cn/problems/longest-valid-parentheses/solutions/314683/zui-chang-you-xiao-gua-hao-by-leetcode-solution/

```java
/**
借助一个栈（Deque）来解决有效括号子串的问题。当遇到左括号时，将其下标入栈；当遇到右括号时，弹出栈顶元素表示匹配成功。对于有效的括号子串，栈中存储的是括号的下标，而不是括号本身
如果我们不事先将 -1 放入栈中，那么当字符串 s 的第一个字符是右括号时，栈将为空，导致后续计算有效括号子串长度时出现问题。通过事先放入 -1，我们可以确保栈中始终有一个元素，从而使得后续计算有效括号子串长度的操作能够正确进行。
*/
class Solution {
    public int longestValidParentheses(String s) {
        int maxLen = 0;
        Deque<Integer> deque = new LinkedList<>(); // 使用栈来存储括号的下标
        deque.push(-1); // 在栈中事先放入-1，以处理特殊情况
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') { // 遇到左括号，将其下标入栈
                deque.push(i);
            } else { // 遇到右括号
                deque.pop(); // 弹出栈顶元素，表示匹配成功
                if (deque.isEmpty()) { // 栈为空，当前右括号没有匹配的左括号与之对应
                    deque.push(i); // 将当前右括号的下标入栈，以便计算后续的有效括号子串长度
                } else { // 栈非空，计算当前有效括号子串的长度
                    maxLen = Math.max(maxLen, i - deque.peek());
                }
            }
        }
        return maxLen;
    }
}
```

## 17.搜索旋转排序数组 Medium

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

**示例 1：**

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**示例 3：**

```
输入：nums = [1], target = 0
输出：-1
```

**提示：**

- `1 <= nums.length <= 5000`
- `-104 <= nums[i] <= 104`
- `nums` 中的每个值都 **独一无二**
- 题目数据保证 `nums` 在预先未知的某个下标上进行了旋转
- `-104 <= target <= 104`

Related Topics

- 数组
- 二分查找

```java
//二分法
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }else if (nums[mid] < nums[right]) {//右半部分有序
                if (nums[mid] < target && target <= nums[right]) {//目标值是否在右半部分
                    left = mid + 1;
                }else {
                    right = mid - 1;
                }
            }else {//左半部分有序
                if (nums[left] <= target && target < nums[mid]) {//目标值是否在左半部分
                    right = mid - 1;
                }else {
                    left = mid + 1;
                }
            }
        }
        return -1;
    }
}
```

## 18.在排序数组中查找元素的第一个和最后一个位置 Medium

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

**示例 1：**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```
输入：nums = [], target = 0
输出：[-1,-1]
```

**提示：**

- `0 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`
- `nums` 是一个非递减数组
- `-109 <= target <= 109`

Related Topics

- 数组
- 二分查找

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        //得到的边界是开区间的
        int leftBorder = findLeftBorder(nums,target);
        int rightBorder = findRightBorder(nums,target);
        //情况一：target小于数组的最小值或大于最大值，那么leftBorder和rightBorder就不会被更新
        if (leftBorder == -2 || rightBorder == -2) {
            return new int[] {-1,-1};
        }
        //情况二：target在数组中
        if (rightBorder - leftBorder > 1){
            return new int[]{leftBorder+1,rightBorder-1};
        }
        //情况三：target介于数组最小值和最大值之间，但不存在于数组中
        return new int[]{-1,-1};

    }

    /**
     * 寻找左边界
     * @param nums
     * @param target
     * @return
     */
    public int findLeftBorder(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        int leftBorder = -2;
        while (left <= right) {
            int mid = left + (right - left) /2;
            if (nums[mid] >= target) {
                right = mid -1;
                leftBorder = right;
            }else if (nums[mid] < target) {
                left = mid + 1;
            }

        }
        return leftBorder;
    }

    /**
     * 寻找右边界
     * @param nums
     * @param target
     * @return
     */
    public int findRightBorder(int[] nums,int target) {
        int left = 0;
        int right = nums.length-1;
        int rightBorder = -2;
        while (left <= right) {
            int mid = left + (right - left) /2;
            if (nums[mid] <= target) {
                left = mid +1;
                rightBorder = left;
            }else if (nums[mid] > target) {
                right = mid - 1;
            }

        }
        return rightBorder;
    }

}
```

## 19.组合总和 Medium

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

**示例 1：**

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
```

**示例 2：**

```
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
```

**示例 3：**

```
输入: candidates = [2], target = 1
输出: []
```

**提示：**

- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- `candidates` 的所有元素 **互不相同**
- `1 <= target <= 40`

Related Topics

- 数组
- 回溯

```java
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    List<Integer> path = new LinkedList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates== null || candidates.length == 0) {
            return res;
        }
        backTracking(candidates,target,0);
        return res;
    }

    public void backTracking(int[] candidates, int target,int index) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            if (target - candidates[i] < 0) continue;
            path.add(candidates[i]);
            backTracking(candidates,target-candidates[i],i);
            path.remove(path.size()-1);
        }
    }
}
```

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return res;
        }

        // 优化1：对候选数组进行排序
        //通过对候选数组进行排序，可以将较小的元素尽早放入组合中，从而在搜索过程中更早地进行剪枝操作。当目标值小于当前选择的数字时，可以提前结束搜索，因为后续的数字更大，无法满足目标值要求。
        Arrays.sort(candidates);

        backTracking(candidates, target, 0);
        return res;
    }

    public void backTracking(int[] candidates, int target, int start) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < candidates.length; i++) {
            // 优化2：目标值的剪枝
            if (target < candidates[i]) {
                break;
            }

            path.add(candidates[i]);
            backTracking(candidates, target - candidates[i], i);
            path.remove(path.size() - 1);

        }
    }
}
```

## 20.接雨水 Hard

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

**示例 2：**

```
输入：height = [4,2,0,3,2,5]
输出：9
```

**提示：**

- `n == height.length`
- `1 <= n <= 2 * 104`
- `0 <= height[i] <= 105`

Related Topics

- 栈
- 数组
- 双指针
- 动态规划
- 单调栈

思路：https://www.bilibili.com/video/BV1Qg411q7ia/?vd_source=92cb5cb9faa01574e9b1f82bf91d080d

```java

```

```java
class Solution {
    public int trap(int[] height) {
        int len = height.length;
        int pre_max = 0;
        int suf_max = 0;
        int left = 0;
        int right = len - 1;
        int res = 0;
        while (left <= right) {
            pre_max = Math.max(pre_max, height[left]);
            suf_max = Math.max(suf_max, height[right]);
            res += pre_max < suf_max ? pre_max - height[left++] : suf_max-height[right--];
        }
        return res;
    }
}
```

























































