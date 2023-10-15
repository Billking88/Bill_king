# 力扣Hot100

## 21.全排列 Medium

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**示例 2：**

```
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

**示例 3：**

```
输入：nums = [1]
输出：[[1]]
```

**提示：**

- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有整数 **互不相同**

Related Topics

- 数组
- 回溯

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();
    boolean[] isUsed;
    public List<List<Integer>> permute(int[] nums) {
        isUsed = new boolean[nums.length];//用于记录当前数字是否被用过
        backTracking(nums);
        return res;
    }

    public void backTracking(int[] nums){
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!isUsed[i]){
                isUsed[i] = true;
                path.add(nums[i]);
                backTracking(nums);
                path.removeLast();
                isUsed[i] = false;
            }
        }
    }
}
```

## 22.旋转图像 Medium

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 **原地** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

**示例 1：**

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**示例 2：**

```
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

**提示：**

- `n == matrix.length == matrix[i].length`
- `1 <= n <= 20`
- `-1000 <= matrix[i][j] <= 1000`

Related Topics

- 数组
- 数学
- 矩阵

解析：https://leetcode.cn/problems/rotate-image/solutions/1228078/48-xuan-zhuan-tu-xiang-fu-zhu-ju-zhen-yu-jobi/

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < (n + 1) / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 -j][i] = matrix[n-1-i][n-1-j];
                matrix[n-1-i][n-1-j] = matrix[j][n -1-i];
                matrix[j][n-1-i] = temp;
            }
        }
    }
}
```

## 23.字母异位词分组 Medium

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

**示例 1:**

```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

**示例 2:**

```
输入: strs = [""]
输出: [[""]]
```

**示例 3:**

```
输入: strs = ["a"]
输出: [["a"]]
```

**提示：**

- `1 <= strs.length <= 104`
- `0 <= strs[i].length <= 100`
- `strs[i]` 仅包含小写字母

Related Topics

- 数组
- 哈希表
- 字符串
- 排序

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> map  = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            //排序
            Arrays.sort(chars);
            //将排序后得到的字符串作为map的key
            String key = new String(chars);
            //通过key查找出list，如果没有当前key对应的list，那就创建一个新的空list
            List<String> list = map.getOrDefault(key,new ArrayList<>());
            //list保存字符串
            list.add(str);
            //map更新key对应的list
            map.put(key,list);
        }
        //根据key进行分组
        return new ArrayList<List<String>>(map.values());
    }
}
```

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> map  = new HashMap<>();
        for (String str : strs) {
            int[] records = new int[26];
            //记录每个字母的数量
            for (int i = 0; i < str.length(); i++) {
                records[str.charAt(i) -'a']++;
            }
            StringBuilder sb = new StringBuilder();
            //只要得到的sb是相同的，那就说明是字母异位词
            for (int i = 0; i < records.length; i++) {
                if (records[i] > 0) {
                    sb.append((char) 'a'+i).append(records[i]);
                }
            }
            //得到的字符串作为map的key
            String key = sb.toString();
            //通过key查找出list，如果没有当前key对应的list，那就创建一个新的空list
            List<String> list = map.getOrDefault(key,new ArrayList<>());
            //list保存字符串
            list.add(str);
            //map更新key对应的list
            map.put(key,list);
        }
        //根据key进行分组
        return new ArrayList<List<String>>(map.values());
    }
}
```

## 24.最大子数组和 Medium

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

**示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**示例 2：**

```
输入：nums = [1]
输出：1
```

**示例 3：**

```
输入：nums = [5,4,-1,7,8]
输出：23
```

**提示：**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`



**进阶：**如果你已经实现复杂度为 `O(n)` 的解法，尝试使用更为精妙的 **分治法** 求解。

Related Topics

- 数组
- 分治
- 动态规划

```java
//动态规划
class Solution {
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(nums[i],dp[i-1]+nums[i]);
        }
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < dp.length; i++) {
            max = Math.max(max,dp[i]);
        }
        return max;
    }
}

```

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        int curSum = nums[0];
        for (int i = 1; i < nums.length; i++) {
            curSum = Math.max(nums[i],curSum+nums[i]);
            maxSum = Math.max(maxSum,curSum);
        }
        return maxSum;
    }
}
```

## 25.跳跃游戏 Medium

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

**提示：**

- `1 <= nums.length <= 104`
- `0 <= nums[i] <= 105`

Related Topics

- 贪心
- 数组
- 动态规划

```java
//贪心算法：最大距离等于当前位置加上对应的nums[i]
class Solution {
    public boolean canJump(int[] nums) {
        int maxDistance = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (i > maxDistance) {//如果当前位置都比最大距离大了，那肯定无法到达
                return false;
            }
            maxDistance = Math.max(maxDistance,i+nums[i]);//更新最大距离
        }
        return true;
    }
}
```

## 26.合并区间 Medium

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

**例 1：**

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

**示例 2：**

```
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

**提示：**

- `1 <= intervals.length <= 104`
- `intervals[i].length == 2`
- `0 <= starti <= endi <= 104`

Related Topics

- 数组
- 排序



```java
class Solution {
    public int[][] merge(int[][] intervals) {
        //讨论特殊情况
        if (intervals.length == 0) {
            return new int[0][2];
        }
        //对二维数组进行排序，根据二维数组中一维数组的第一个数字从小到大进行排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        //因为不知道有多少一维数组需要合并，所以最终得到的二维数组的长度也不知道，所以就创建一个list来存储一维数组
        List<int[]> list = new ArrayList<>();
        //将排序后的第一个一维数组放进list，遍历剩下的一维数组
        list.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            //当前数组的左值
            int curArrLeft = intervals[i][0];
            //当前数组的右值
            int curArrRight = intervals[i][1];
            //列表中的最后一个数组的左值
            int listFinalArrLeft = list.get(list.size() -1)[0];
            //列表中的最后一个数组的右值
            int listFinalArrRight = list.get(list.size() - 1)[1];
            if (curArrLeft <= listFinalArrRight) {//如果当前数组左值小于了列表中最后一个数组的右值，说明有重叠
                //修改列表中的最后一个数组的右值
                list.get(list.size()-1)[1] = Math.max(curArrRight,listFinalArrRight);
            }else {
                //没有重合就将当前的一维数组放入列表中
                list.add(new int[]{intervals[i][0],intervals[i][1]});
            }

        }
        return list.toArray(new int[list.size()][]);
    }
}
```

## 27.不同路径 Medium

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

**示例 1：**

```
输入：m = 3, n = 7
输出：28
```

**示例 2：**

```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```

**示例 3：**

```
输入：m = 7, n = 3
输出：28
```

**示例 4：**

```
输入：m = 3, n = 3
输出：6
```

**提示：**

- `1 <= m, n <= 100`
- 题目数据保证答案小于等于 `2 * 109`

Related Topics

- 数学
- 动态规划
- 组合数学

```java
//动态规划
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        //初始化
        //第一行，每个格子都只有一种到达方法
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        //列也是
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }
        //遍历
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                //递推公式
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
}
```

## 28.最小路径和

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

**示例 1：**

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 200`
- `0 <= grid[i][j] <= 200`

Related Topics

- 数组
- 动态规划
- 矩阵

这题跟27题很类似，思路是差不多的

```java
class Solution {
    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        //列
        for (int i = 1; i < grid.length; i++) {
            dp[i][0] = grid[i][0]+dp[i-1][0];
        }
        //行
        for (int j = 1; j < grid[0].length; j++) {
            dp[0][j] = grid[0][j]+dp[0][j-1];
        }
        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[0].length; j++) {
                dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[grid.length-1][grid[0].length-1];
    }
}
```



## 29.爬楼梯 Easy

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**示例 1：**

```
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

**示例 2：**

```
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

**提示：**

- `1 <= n <= 45`

Related Topics

- 记忆化搜索
- 数学
- 动态规划



```java
class Solution {
    public int climbStairs(int n) {
        //dp[i]表示爬到第i阶楼梯时有多少种方法
        int[] dp = new int[n+1];
        //初始化
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            //递推公式
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
}
```

## 30.编辑距离 Hard

给你两个单词 `word1` 和 `word2`， *请返回将 word1 转换成 word2 所使用的最少操作数* 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

**提示：**

- `0 <= word1.length, word2.length <= 500`
- `word1` 和 `word2` 由小写英文字母组成

Related Topics

- 字符串
- 动态规划

解析：https://www.programmercarl.com/0072.%E7%BC%96%E8%BE%91%E8%B7%9D%E7%A6%BB.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE

```java
class Solution {
    public int minDistance(String word1, String word2) {
		//dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]。
        int[][] dp = new int[word1.length() +1][word2.length() + 1];
        //这里可以理解为word2为空字符串，遍历到word的第i个自负，word2就需要增加i次才能变成跟word1一样
        for (int i = 1; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= word2.length(); j++) {
            dp[0][j] = j;
        }
		
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {、
                //如果当前两个字符相等，那么最短编辑距离跟dp[i-1][j-1]相等
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    //删(增)：dp[i-1][j]+1，dp[i][j-1]+1.删或增最终的结果是一样的
                    //改：dp[i-1][j-1]+1
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                }

            }
        }
        return dp[word1.length()][word2.length()];
    }
}
```

## 31.颜色分类 Medium

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

**示例 1：**

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

**示例 2：**

```
输入：nums = [2,0,1]
输出：[0,1,2]
```

**提示：**

- `n == nums.length`
- `1 <= n <= 300`
- `nums[i]` 为 `0`、`1` 或 `2`

**进阶：**

- 你能想出一个仅使用常数空间的一趟扫描算法吗？

Related Topics

- 数组
- 双指针
- 排序

方法一：逐一覆盖，用idea模拟一遍就清楚思路了

```java
class Solution {
    public void sortColors(int[] nums) {
        int zeroIndex = 0;
        int oneIndex = 0;
        for (int i = 0; i < nums.length; i++) {
            int currentNum = nums[i];
            nums[i] = 2;
            if (currentNum < 2) {
                nums[oneIndex++] = 1;
            }
            if (currentNum < 1) {
                nums[zeroIndex++] = 0;
            }
        }
    }
}
```

方法二：替换，思路是遇到0就跟左边界的元素替换，遇到2就跟右边界的元素替换

```java
class Solution {
    public void sortColors(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        int current = 0;//当前指向元素的索引
        while (current <= right){//终止条件是当前索引>右边界
            if (nums[current] == 0) {
                swap(nums,current,left);
                left++;
                current++;
            }else if (nums[current] == 2) {
                swap(nums,current,right);
                right--;
                //在这里不需要current++，因为替换之后的元素还得再继续比较
            }else {
                current++;
            }
        }
    }

    public void swap(int[] nums, int left,int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }

}
```

## 32.最小覆盖子串 Hard

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

**注意：**

- 对于 `t` 中重复字符，我们寻找的子字符串中该字符数量必须不少于 `t` 中该字符数量。
- 如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

**示例 1：**

```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
```

**示例 2：**

```
输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。
```

**示例 3:**

```
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
```

**提示：**

- `m == s.length`
- `n == t.length`
- `1 <= m, n <= 105`
- `s` 和 `t` 由英文字母组成

进阶：

你能设计一个在 o(m+n)时间内解决此问题的算法吗？

Related Topics

- 哈希表
- 字符串
- 滑动窗口



```java
class Solution {
    public String minWindow(String s, String t) {
        //arr用于记录字符串中的字符需要的个数，用ASCII玛作为索引
        int[] arr = new int[128];
        for (int i = 0; i < t.length(); i++) {
            arr[t.charAt(i)]++;
        }
        //滑动窗口左右边界
        int left = 0, right = 0;
 		//size表示窗口的长度
        int size = Integer.MAX_VALUE;
        //start用于记录窗口的起始位置，这里不用left记录，因为left可能随时更新
        int start = 0;
        //count表示在遍历s的时候，还需要多少个t中要求的字符
        int count = t.length();
        while (right < s.length()) {
            char c = s.charAt(right);
            //如果当前字符的ASCII玛能够在arr中能找到，那就让count-1
            if (arr[c] > 0) {
                count--;
            }
            //无论当前字符是否在t中，我们都让这个字符的ASCII玛在arr中对应的元素-1，哪怕是负数，负数可以认为是这个字符是多余的，后面就可以通过更新left来去除
            arr[c]--;
            //count等于0说明t的所有字符都包含在了滑动窗口中
            if (count == 0) {
                //缩小滑动窗口
                //如果左边界的ASCII玛在arr中对应的值是小于0的，说明这个字符多余，让左边界右移
                while (left < right && arr[s.charAt(left)] < 0) {
                    arr[s.charAt(left)]++;
                    left++;
                }
                //如果这次所求的滑动窗口的大小比上一次满足条件的滑动窗口的大小更小，就更新窗口大小，并更新start的值
                if ((right - left + 1) < size) {
                    size = right - left + 1;
                    start = left;
                }
                //因为count等于0了，那我们就尝试让左边界右移，缩小窗口
                arr[s.charAt(left)]++;
                left++;
                //count也需要加1，因为当前left对应的字符正式t中需要的
                count++;
            }
            //右边界右移
            right++;
        }
        return size == Integer.MAX_VALUE ? "" : s.substring(start,start+size);
    }
}
```

## 33.子集 Medium

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

**提示：**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有元素 **互不相同**

Related Topics

- 位运算
- 数组
- 回溯



```java
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public List<List<Integer>> subsets(int[] nums) {
        dfs(nums,0);
        return res;
    }

    private void dfs(int[] nums,int index){
        res.add(new ArrayList<>(path));
        if (index == nums.length) {
            return;
        }

        for (int i = index; i < nums.length; i++) {
            path.add(nums[i]);
            dfs(nums,i+1);
            path.removeLast();
        }
    }


}
```



## 34.单词搜索 Medium

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例 1：**

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
```

**示例 3：**

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

**提示：**

- `m == board.length`
- `n = board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- `board` 和 `word` 仅由大小写英文字母组成

**进阶：**你可以使用搜索剪枝的技术来优化解决方案，使其在 `board` 更大的情况下可以更快解决问题？

Related Topics

- 数组
- 回溯
- 矩阵



```java
class Solution {
	//将board的长宽作为全局变量
    int m;
    int n;
    public boolean exist(char[][] board, String word) {
        this.m = board.length;
        this.n = board[0].length;
        char[] chars = word.toCharArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (backTracking(board,i,j,chars,0)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean backTracking(char[][] board,int i, int j, char[] chars,int index) {
        //如果index跟chars长度相等，说明找到了单词
        if (index == chars.length) {
            return true;
        }
        //判断越界
        if (i < 0 || j < 0 || i >=m || j >= n) {
            return false;
        }
		//如果当前遍历的字符跟word中要求的不一样，false
        if (board[i][j] != chars[index]) {
            return false;
        }
		//用临时变量记录board[i][j]，然后修改为别的字符，这样就算再返回来遍历这个字符，也不会被重复使用
        char temp = board[i][j];
        board[i][j] = '#';
		//递归相邻的字符
        boolean res =backTracking(board,i-1,j,chars,index+1)
                || backTracking(board,i,j-1,chars,index+1)
                || backTracking(board,i+1,j,chars,index+1)
                || backTracking(board,i,j+1,chars,index+1);
        //记得恢复原来的字符
        board[i][j] = temp;
        return res;
    }
}
```



## 35.柱状图中最大的矩形 Hard

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4
```

**提示：**

- `1 <= heights.length <=105`
- `0 <= heights[i] <= 104`

Related Topics

- 栈
- 数组
- 单调栈

文字解析：https://www.programmercarl.com/0084.%E6%9F%B1%E7%8A%B6%E5%9B%BE%E4%B8%AD%E6%9C%80%E5%A4%A7%E7%9A%84%E7%9F%A9%E5%BD%A2.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC

视频解析：https://www.bilibili.com/video/BV1Ns4y1o7uB/?vd_source=92cb5cb9faa01574e9b1f82bf91d080d

```java
//这里就不写注解了，不太好描述思路，思路详见文字解析或视频解析
class Solution {
    public int largestRectangleArea(int[] heights) {
        int maxArea = 0;
        int[] newHeights= new int[heights.length + 2];
        newHeights[0] = 0;
        newHeights[newHeights.length-1] = 0;
        for (int i = 0; i < heights.length; i++) {
            newHeights[i+1] = heights[i];
        }
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        for (int i = 1 ;i < newHeights.length; i++) {
            if (newHeights[i] > newHeights[stack.peek()]) {
                stack.push(i);
            }else if (newHeights[i] == newHeights[stack.peek()]) {
                stack.pop();
                stack.push(i);
            }else {
                while (newHeights[i] < newHeights[stack.peek()]) {
                    int mid = stack.pop();
                    int left = stack.peek();
                    int right = i;
                    int width = right - left -1;
                    int height = newHeights[mid];
                    maxArea = Math.max(maxArea,width * height);
                }
                stack.push(i);
            }
        }
        return maxArea;
    }

}
```

精简版：

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int maxArea = 0;
        int[] newHeights= new int[heights.length + 2];
        newHeights[0] = 0;
        newHeights[newHeights.length-1] = 0;
        for (int i = 0; i < heights.length; i++) {
            newHeights[i+1] = heights[i];
        }

        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0 ;i < newHeights.length; i++) {
            while (!stack.isEmpty() && newHeights[i] < newHeights[stack.peek()]) {
                int mid = stack.pop();
                int left = stack.peek();
                int right = i;
                int width = right - left -1;
                int height = newHeights[mid];
                maxArea = Math.max(maxArea,width * height);
            }
            stack.push(i);

        }
        return maxArea;
    }
}
```

## 36.最大矩形 Hard 

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

**示例 1：**

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```

**示例 2：**

```
输入：matrix = []
输出：0
```

**示例 3：**

```
输入：matrix = [["0"]]
输出：0
```

**示例 4：**

```
输入：matrix = [["1"]]
输出：1
```

**示例 5：**

```
输入：matrix = [["0","0"]]
输出：0
```

**提示：**

- `rows == matrix.length`
- `cols == matrix[0].length`
- `1 <= row, cols <= 200`
- `matrix[i][j]` 为 `'0'` 或 `'1'`

Related Topics

- 栈
- 数组
- 动态规划
- 矩阵
- 单调栈

文字解析：https://leetcode.cn/problems/maximal-rectangle/solutions/1774721/by-burling-lucky-gqg0/

这道题跟上面那一题挺像的

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        if (m == 0) {
            return 0;
        }

        List<int[]> allRowsHeights = new ArrayList<>();
        for (int i = m-1; i >= 0 ; i--) {
            int[] height = new int[n + 2];
            for (int j = 0; j < n; j++) {
                int k = i;
                while (k >= 0 && matrix[k][j] == '1') {
                    height[j+1]++;
                    k--;
                }
            }
            allRowsHeights.add(height);
        }

        int maxArea = 0;
        for(int[] heights:allRowsHeights) {
            Deque<Integer> stack = new ArrayDeque<>();
            for (int i = 0; i < heights.length; i++) {
                while (!stack.isEmpty() && heights[stack.peek()] > heights[i]) {
                    int cur = stack.pop();
                    int val = heights[cur] * (i - stack.peek() - 1);
                    maxArea = Math.max(maxArea, val);
                }
                stack.push(i);

            }
        }
        return maxArea;
    }
}
```

## 37.二叉树的中序遍历 Easy

给定一个二叉树的根节点 `root` ，返回 *它的 \**中序** 遍历* 。

**示例 1：**

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：root = []
输出：[]
```

**示例 3：**

```
输入：root = [1]
输出：[1]
```

**提示：**

- 树中节点数目在范围 `[0, 100]` 内
- `-100 <= Node.val <= 100`

**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？

Related Topics

- 栈
- 树
- 深度优先搜索
- 二叉树

```java
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        dfs(root);
        return res;
    }


    public void dfs(TreeNode root) {
        if (root == null) {
            return ;
        }

        inorderTraversal(root.left);
        res.add(root.val);
        inorderTraversal(root.right);
    }

}
```

## 38.不同的二叉搜索树 Medium

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

**示例 1：**

```
输入：n = 3
输出：5
```

**示例 2：**

```
输入：n = 1
输出：1
```

**提示：**

- `1 <= n <= 19`

Related Topics

- 树
- 二叉搜索树
- 数学
- 动态规划
- 二叉树

解析：https://www.programmercarl.com/0096.%E4%B8%8D%E5%90%8C%E7%9A%84%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91.html

```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }
}
```

## 39.验证二叉搜索树 Medium

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例 1：**

```
输入：root = [2,1,3]
输出：true
```

**示例 2：**

```
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
```

**提示：**

- 树中节点数目范围在`[1, 104]` 内
- `-231 <= Node.val <= 231 - 1`

Related Topics

- 树
- 深度优先搜索
- 二叉搜索树
- 二叉树

```java

//思路：二叉搜索树的中序遍历是有序的，所以只要按中序遍历记录每个节点的值，然后判断是不是有序即可
class Solution {

    public boolean isValidBST(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        dfs(root,list);
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i) <= list.get(i-1)) {
                return false;
            }
        }
        return true;
    }


    public void dfs(TreeNode root, List<Integer> list) {
        if (root == null) {
            return ;
        }
        dfs(root.left,list);
        list.add(root.val);
        dfs(root.right,list);
    }

}
```



思路2：在遍历的时候就进行比较

```java
class Solution {
    TreeNode pre = null;//记录上一个节点
    public boolean isValidBST(TreeNode root) {

        if (root == null) {//空节点也是二叉搜索树
            return true;
        }

        boolean left = isValidBST(root.left); //左边是不是二叉搜索树

        if (pre != null && pre.val >= root.val) {
            return false;
        }

        boolean right = isValidBST(root.right);//右边是不是二叉搜索树

        return left && right;
        
    }
    
}
```

## 40.对称二叉树 Easy

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

**示例 1：**

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2：**

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

**提示：**

- 树中节点数目在范围 `[1, 1000]` 内
- `-100 <= Node.val <= 100`



**进阶：**你可以运用递归和迭代两种方法解决这个问题吗？

Related Topics

- 树
- 深度优先搜索
- 广度优先搜索
- 二叉树

```java
//递归
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return compare(root.left,root.right);
    }

    public boolean compare(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left != null && right == null) {
            return false;
        }

        if (left == null && right != null) {
            return false;
        }

        if (left.val != right.val) {
            return false;
        }

        return compare(left.left,right.right) && compare(left.right,right.left);

    }

}
```

```java
//迭代
class Solution {
    public boolean isSymmetric(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerFirst(root.left);
        deque.offerLast(root.right);
        while (!deque.isEmpty()) {
            TreeNode leftNode = deque.pollFirst();
            TreeNode rightNode = deque.pollLast();
            if (leftNode == null && rightNode == null) {
                continue;
            }

            if (leftNode != null && rightNode == null) {
                return false;
            }

            if (leftNode == null && rightNode != null) {
                return false;
            }
            if (leftNode.val != rightNode.val ) {
                return false;
            }

            deque.offerFirst(leftNode.left);
            deque.offerLast(rightNode.right);
            deque.offerFirst(leftNode.right);
            deque.offerLast(rightNode.left);

        }

        return true;

    }


}
```