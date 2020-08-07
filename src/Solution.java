import java.util.*;

public class Solution {
    //JZ-01. 二维数组中的的查找
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length < 1 || array[0].length < 1) {
            return false;
        }
        int i = 0;
        int j = array[0].length - 1;
        while (i < array.length && j >= 0) {
            if (array[i][j] < target) {
                i++;
            } else if (array[i][j] > target) {
                j--;
            } else {
                return true;
            }
        }
        return false;
    }

    //JZ-02. 替换空格
    public String replaceSpace(StringBuffer str) {
        int cnt = 0;
        int i;
        for (i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                cnt++;
            }
        }
        i--;
        str.setLength(str.length() + cnt * 2);
        int j = str.length() - 1;
        while (i >= 0 && j >= 0) {
            if (str.charAt(i) != ' ') {
                str.setCharAt(j--, str.charAt(i--));
            } else {
                str.setCharAt(j--, '0');
                str.setCharAt(j--, '2');
                str.setCharAt(j--, '%');
                i--;
            }
        }
        return str.toString();
    }

    //JZ-03. 从尾到头打印链表
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> ret = new ArrayList<>();
        printListFromTailToHead(listNode, ret);
        return ret;
    }

    public void printListFromTailToHead(ListNode listNode, ArrayList<Integer> ret) {
        if (listNode != null) {
            printListFromTailToHead(listNode.next, ret);
            ret.add(listNode.val);
        }
    }

    //JZ-04. 重建二叉树
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        TreeNode root = null;
        Stack<TreeNode> stack = new Stack<>();
        int i = 0, j = 0;

        while (!stack.isEmpty() || i < pre.length || j < in.length) {
            if (stack.isEmpty()) {
                stack.push(new TreeNode(pre[i++]));
                if (root == null) {
                    root = stack.peek();
                }
            } else if (stack.peek().val == in[j]) {
                TreeNode node = stack.pop();
                j++;
                if (i >= pre.length) {
                    continue;
                }
                if (stack.isEmpty() || stack.peek().val != in[j]) {
                    node.right = new TreeNode(pre[i++]);
                    stack.push(node.right);
                }
            } else {
                stack.peek().left = new TreeNode(pre[i++]);
                stack.push(stack.peek().left);
            }
        }
        return root;
    }

    //JZ-05.用两个栈实现队列
    private Stack<Integer> stack1 = new Stack<>();
    private Stack<Integer> stack2 = new Stack<>();

    /*因为会和JZ-20产生函数名冲突。故注释掉
        public void push(int node) {
            stack2.push(node);
        }

        public int pop() {
            if (stack1.isEmpty()) {
                while (!stack2.isEmpty()) {
                    stack1.push(stack2.pop());
                }
            }
            return stack1.pop();
        }
    */
    //JZ-06.旋转数组的最小数字
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length < 1) {
            return 0;
        }
        int left, mid, right;
        left = 0;
        right = array.length - 1;
        while (left < right) {
            mid = (left + right) / 2;
            if (array[mid] > array[right]) {
                left = mid + 1;
            } else if (array[mid] < array[right]) {
                right = mid;
            } else {
                right = mid;
            }
        }
        return array[left];
    }

    //JZ-07.斐波那契数列
    public int Fibonacci(int n) {
        if (n < 2) {
            return n;
        }
        int[] f = new int[n + 1];
        f[0] = 0;
        f[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }

    //JZ-08.跳台阶
    //f(n)=f(n-1)+f(n-2)
    public int JumpFloor(int target) {
        return Fibonacci(target + 1);
    }

    //JZ-09变态跳台阶
    //f(n)=1+f(n-1)+f(n-2)...f(1);
    //f(n-1)=1+f(n-2)+...f(1);
    //f(n)=f(n-2)*2
    public int JumpFloorII(int target) {
        if (target < 1) {
            return 1;
        }
        return 1 << (target - 1);
    }

    //JZ-10.矩形覆盖
    //f(n)=f(n-1)+f(n-2)
    public int RectCover(int target) {
        if (target < 2) {
            return target;
        }
        return Fibonacci(target + 1);
    }

    //JZ-11.二进制中1的个数
    public int NumberOf1(int n) {
        int ret = 0;
        while (n != 0) {
            ret += n & 1;
            n = n >>> 1;
        }
        return ret;
    }

    //JZ-12。数值的整数次方
    public double Power(double base, int exponent) {
        if (base == 0) {
            return 0;
        }
        if (exponent == 0) {
            return 1;
        }
        boolean isNeg = false;
        double ret = 1;
        if (exponent < 0) {
            exponent = -exponent;
            isNeg = true;
        }

        while (exponent != 0) {
            if ((exponent & 1) != 0) {
                ret *= base;
            }
            exponent = exponent >> 1;
            base = base * base;
        }
        if (isNeg) {
            ret = 1 / ret;
        }
        return ret;
    }

    //JZ-13.调整数组顺序使奇数位于偶数前面
    //辅助数组：空间O(n)、时间O(n)
    //in-place：空间O(1)、时间O(n^2)
    public void reOrderArray(int[] array) {
        for (int i = array.length - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) {
                if (array[j] % 2 == 0 && array[j + 1] % 2 == 1) {
                    int t = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = t;
                }
            }
        }
    }

    //JZ-14.链表中的倒数第k个结点
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode slow = head, fast = head;
        while (k > 0) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
            k--;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    //JZ-15.反转链表
    public ListNode ReverseList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode node = head;
        head = head.next;
        node.next = null;
        while (head != null) {
            ListNode t = head;
            head = head.next;
            t.next = node;
            node = t;
        }
        return node;
    }

    //JZ-16.合并两个排序的链表
    public ListNode Merge(ListNode list1, ListNode list2) {
        ListNode list3 = new ListNode(0);
        ListNode head = list3;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                list3.next = list1;
                list1 = list1.next;
            } else {
                list3.next = list2;
                list2 = list2.next;
            }
            list3 = list3.next;
        }
        if (list1 != null) {
            list3.next = list1;
        } else {
            list3.next = list2;
        }
        return head.next;
    }

    //JZ-17.树的子结构
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root2 == null || root1 == null) {
            return false;
        }

        return f(root1, root2)
                || HasSubtree(root1.left, root2)
                || HasSubtree(root1.right, root2);
    }

    //判断是否是同根节点的子结构
    public boolean f(TreeNode r1, TreeNode r2) {
        if (r1 == r2) {
            return true;
        }
        if (r1 == null) {
            return false;
        }
        if (r2 == null) {
            return true;
        }

        if (r1.val != r2.val) {
            return false;
        } else {
            return f(r1.left, r2.left) && f(r1.right, r2.right);
        }
    }

    //JZ-18.二叉树的镜像
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            root = queue.poll();
            if (root.left != null) {
                queue.offer(root.left);
            }
            if (root.right != null) {
                queue.offer(root.right);
            }
            TreeNode t = root.left;
            root.left = root.right;
            root.right = t;
        }
    }

    //JZ-19.顺时针打印矩阵
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        int dir = 0;//方向 0-右 1-下 2-左 3-上
        int times = 0;//方向尝试次数
        int i = 0, j = -1;//指针
        int rows = matrix.length, cols = matrix[0].length;//size
        boolean[][] visited = new boolean[rows][cols];
        ArrayList<Integer> retList = new ArrayList<>();
        while (times < 4) {
            switch (dir) {
                case 0:
                    if (j + 1 < cols && !visited[i][j + 1]) {
                        times = 0;
                        j++;
                        visited[i][j] = true;
                        retList.add(matrix[i][j]);
                    } else {
                        times++;
                        dir = (dir + 1) % 4;
                    }
                    break;
                case 1:
                    if (i + 1 < rows && !visited[i + 1][j]) {
                        times = 0;
                        i++;
                        visited[i][j] = true;
                        retList.add(matrix[i][j]);
                    } else {
                        times++;
                        dir = (dir + 1) % 4;
                    }
                    break;
                case 2:
                    if (j > 0 && !visited[i][j - 1]) {
                        times = 0;
                        j--;
                        visited[i][j] = true;
                        retList.add(matrix[i][j]);
                    } else {
                        times++;
                        dir = (dir + 1) % 4;
                    }
                    break;
                default:
                    if (i > 0 && !visited[i - 1][j]) {
                        times = 0;
                        i--;
                        visited[i][j] = true;
                        retList.add(matrix[i][j]);
                    } else {
                        times++;
                        dir = (dir + 1) % 4;
                    }
                    break;
            }
        }
        return retList;
    }

    //JZ-20.包含min函数的栈
    Stack<Integer> stack = new Stack<>();
    Stack<Integer> minStack = new Stack<>();

    public void push(int node) {
        if (!stack.isEmpty()) {
            node = Math.min(stack.peek(), node);
        }
        stack.push(node);
        minStack.push(node);
    }

    public void pop() {
        stack.pop();
        minStack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int min() {
        return minStack.peek();
    }

    //JZ-21.栈的压入弹出序列
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        Stack<Integer> stack = new Stack<>();
        int i = 0, j = 0;
        if (pushA.length != popA.length) {
            return false;
        }
        while (j < popA.length) {
            if (stack.isEmpty()) {
                stack.push(pushA[i++]);
            } else if (stack.peek() != popA[j]) {
                if (i >= pushA.length) {
                    return false;
                }
                stack.push(pushA[i++]);
            } else {
                stack.pop();
                j++;
            }
        }
        return true;
    }

    //JZ-22.从上往下打印二叉树
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        ArrayList<Integer> retList = new ArrayList<>();
        if (root != null) {
            queue.offer(root);
        }
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            retList.add(node.val);
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        return retList;
    }

    //JZ-23.二叉搜索树的后序遍历序列
    public boolean VerifySquenceOfBST(int[] sequence) {
        //return f1(sequence);
        return f2(sequence);
    }

    //O（n）
    public boolean f1(int[] sequence) {
        if (sequence == null || sequence.length < 1) {
            return false;
        }
        if (sequence.length < 2) {
            return true;
        }

        int i = sequence.length - 1;
        Stack<Integer> stack = new Stack<>();
        int max = Integer.MAX_VALUE;

        stack.push(sequence[i--]);
        while (i >= 0) {
            if (sequence[i] > max) {
                return false;
            }
            if (sequence[i] > sequence[i + 1]) {
                stack.push(sequence[i--]);
            } else {
                while (!stack.isEmpty() && stack.peek() > sequence[i]) {
                    max = stack.pop();
                }
                stack.push(sequence[i--]);
            }
        }
        return true;
    }

    //O（nlogn）
    public boolean f2(int[] sequence) {
        if (sequence == null || sequence.length < 1) {
            return false;
        }
        if (sequence.length < 2) {
            return true;
        }
        return f2(sequence, 0, sequence.length - 1);
    }

    public boolean f2(int[] sequence, int start, int end) {
        if (sequence == null || sequence.length < 1) {
            return false;
        }
        if (sequence.length < 2) {
            return true;
        }
        if (start >= end) {
            return true;
        }
        int k = sequence[end];
        int i = start, j;
        while (sequence[i] < k) {
            i++;
        }
        j = i;
        while (sequence[j] > k) {
            j++;
        }
        if (j != end) {
            return false;
        }
        return f2(sequence, start, i - 1) && f2(sequence, i, end - 1);
    }

    //JZ-24.二叉树中和为某一值得路径
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        //return f1(root, target);
        return f2(root, target);
    }

    //递归
    public ArrayList<ArrayList<Integer>> f1(TreeNode root, int target) {
        ArrayList<Integer> path = new ArrayList<>();
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (root == null) {
            return ret;
        }
        f1(root, target, path, ret);
        return ret;
    }

    public void f1(TreeNode root, int target, ArrayList<Integer> path, ArrayList<ArrayList<Integer>> ret) {
        path.add(root.val);
        target -= root.val;
        if (root.left == null && root.right == null && target == 0) {
            ret.add((ArrayList<Integer>) path.clone());
        }
        if (root.left != null) {
            f1(root.left, target, path, ret);
        }
        if (root.right != null) {
            f1(root.right, target, path, ret);
        }
        path.remove(path.size() - 1);
    }

    //非递归
    public ArrayList<ArrayList<Integer>> f2(TreeNode root, int target) {
        Stack<TreeNode> stack = new Stack<>();
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        ArrayList<Integer> path = new ArrayList<>();

        if (root == null) {
            return ret;
        }
        stack.push(root);
        path.add(root.val);
        target -= root.val;
        root = root.left;
        while (!stack.isEmpty()) {
            if (root != null) {
                stack.push(root);
                path.add(root.val);
                target -= root.val;
                root = root.left;
            } else if (stack.peek().right != null) {
                root = stack.peek().right;
            } else {
                if (stack.peek().left == null && target == 0) {
                    ret.add((ArrayList<Integer>) path.clone());
                }
                while (!stack.isEmpty() && root == stack.peek().right) {
                    root = stack.pop();
                    path.remove(path.size() - 1);
                    target += root.val;
                }
                if (!stack.isEmpty()) {
                    root = stack.peek().right;
                }
            }
        }
        return ret;
    }

    //JZ-25复杂链表的复制
    public RandomListNode Clone(RandomListNode pHead) {
        return Clone_1(pHead);
        //return Clone_2(pHead);
    }

    /*
     *方法一：
     * 在原链表中每个节点后面复制一个节点，然后修改复制节点的指针，最后分离链表
     *     * */
    public RandomListNode Clone_1(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }

        RandomListNode head = pHead;
        while (head != null) {
            RandomListNode node = new RandomListNode(head.label);
            node.random = head.random;
            node.next = head.next;
            head.next = node;
            head = node.next;
        }
        head = pHead.next;
        while (head != null) {
            if (head.random != null) {
                head.random = head.random.next;
            }
            head = head.next;
            if (head != null) {
                head = head.next;
            }
        }
        //注意不能破坏原链表的结构！！！否则会错
        head = pHead;
        pHead = pHead.next;
        while (head.next != null) {
            RandomListNode node = head.next;
            head.next = head.next.next;
            head = node;
        }
        return pHead;
    }

    /*
     * 方法二：
     *建立一个原链表节点和新链表节点的HashMap，通过HashMap寻找相应的节点
     * */
    public RandomListNode Clone_2(RandomListNode pHead) {
        Map<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode node = pHead;
        while (node != null) {
            map.put(node, new RandomListNode(node.label));
            node = node.next;
        }
        node = pHead;
        while (node != null) {
            RandomListNode t = map.get(node);
            t.next = map.get(node.next);
            t.random = map.get(node.random);
            node = node.next;
        }
        return map.get(pHead);
    }

    //JZ-26.二叉搜索树与双向链表
    public TreeNode Convert(TreeNode pRootOfTree) {
        return Convert_1(pRootOfTree);
        //return Convert_2(pRootOfTree);
    }

    //递归
    public TreeNode Convert_1(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        TreeNode[] left = new TreeNode[1];
        TreeNode[] right = new TreeNode[1];
        f(pRootOfTree, left, right);
        return left[0];
    }

    public void f(TreeNode root, TreeNode[] left, TreeNode right[]) {
        TreeNode tmp = null;
        if (root.left != null) {
            f(root.left, left, right);
            root.left = right[0];
            right[0].right = root;
            tmp = left[0];
        } else {
            root.left = null;
        }

        right[0] = root;
        if (root.right != null) {
            f(root.right, left, right);
            root.right = left[0];
            left[0].left = root;
        } else {
            root.right = null;
        }
        left[0] = tmp == null ? root : tmp;
    }

    //非递归
    public TreeNode Convert_2(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = pRootOfTree;
        TreeNode head = null;
        TreeNode tail = null;
        stack.push(node);
        node = node.left;
        while (!stack.isEmpty()) {
            if (node != null) {
                stack.push(node);
                node = node.left;
            } else {
                node = stack.pop();
                if (head == null) {
                    head = node;
                    head.left = null;
                    tail = node;
                } else {
                    tail.right = node;
                    node.left = tail;
                    tail = node;
                }
                node = node.right;
                tail.right = null;
            }
        }
        return head;
    }

    //JZ-27.字符串的排列
    /*
     * 最小的是升序排列-最大的是降序排列
     * 从最小迭代到最大即可
     * */
    public ArrayList<String> Permutation(String str) {
        char[] buf = str.toCharArray();
        ArrayList<String> ret = new ArrayList<>();
        if (str == null || str.length() < 1) {
            return ret;
        }

        Arrays.sort(buf);
        ret.add(new String(buf));
        while (next(buf)) {
            ret.add(new String(buf));
        }
        return ret;
    }

    public boolean next(char[] buf) {
        int k = buf.length - 2;
        while (k >= 0) {
            if (buf[k] < buf[k + 1]) {
                break;
            }
            k--;
        }
        if (k < 0) {
            return false;
        }
        int i = buf.length - 1;
        while (i > k) {
            if (buf[i] > buf[k]) {
                break;
            }
            i--;
        }
        if (i <= k) {
            return false;
        }
        char t = buf[i];
        buf[i] = buf[k];
        buf[k] = t;
        int j = buf.length - 1;
        i = k + 1;
        while (i < j) {
            t = buf[i];
            buf[i] = buf[j];
            buf[j] = t;
            i++;
            j--;
        }
        return true;
    }

    //JZ-28.数组中出现次数超过一半的数字
    public int MoreThanHalfNum_Solution(int[] array) {
//return MoreThanHalfNum_Solution_1(array);
        return MoreThanHalfNum_Solution_2(array);
    }

    /*
     * 方法一：Hash记录次数 时间：O（n）空间：O（n）
     * */
    public int MoreThanHalfNum_Solution_1(int[] array) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = array.length / 2;
        for (int x : array) {
            if (map.containsKey(x)) {
                int freq = map.get(x);
                if (freq >= n) {
                    return x;
                }
                map.put(x, map.get(x) + 1);
            } else {
                map.put(x, 1);
            }
        }
        return 0;
    }

    /*
     * 方法二：in-place
     * 凡是遇到x!=y就将x和y都消去，如果存在要求的数，则最后剩下的一定是所需要的数。
     * */
    public int MoreThanHalfNum_Solution_2(int[] array) {
        if (array == null || array.length < 1) {
            return 0;
        }
        if (array.length == 1) {
            return array[0];
        }
        int cur = 0;
        int cnt = 0;
        for (int i = 0; i < array.length; i++) {
            if (cnt == 0) {
                cur = array[i];
                cnt++;
            } else if (array[i] == cur) {
                cnt++;
            } else {
                cnt--;
            }
        }
        cnt = 0;
        for (int x : array) {
            if (x == cur) {
                cnt++;
                if (cnt > array.length / 2) {
                    return x;
                }
            }
        }
        return 0;
    }

    //JZ-29.最小的k个数
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> ret = new ArrayList<>();
        PriorityQueue<Integer> queue = new PriorityQueue<>(((o1, o2) -> {
            return o2 - o1;
        }));
        if (input == null || input.length < k || k < 1) {
            return ret;
        } else {
            int i;
            for (i = 0; i < k; i++) {
                queue.add(input[i]);
            }
            while (i < input.length) {
                if (input[i] < queue.peek()) {
                    queue.remove();
                    queue.add(input[i]);
                }
                i++;
            }
            ret.addAll(queue);
            return ret;
        }
    }

    //JZ-30.连续子数组的最大和
    public int FindGreatestSumOfSubArray(int[] array) {
        //return FindGreatestSumOfSubArray_1(array);
        return FindGreatestSumOfSubArray_2(array);
    }

    /*方法一：动态规划
     * f(n)表示以n结尾的最大值
     * f(n)=max{f(n-1),0}+arr[n];
     * ret=max{f(0),(1)...f(len-1)};
     * */
    public int FindGreatestSumOfSubArray_1(int[] array) {
        if (array == null || array.length < 1) {
            return 0;
        }
        if (array.length < 2) {
            return array[0];
        }
        int[] f = new int[array.length];
        int ret = 0;
        f[0] = array[0];
        ret = f[0];
        for (int i = 1; i < array.length; i++) {
            f[i] = Math.max(0, f[i - 1]) + array[i];
            ret = Math.max(ret, f[i]);
        }
        return ret;
    }

    /*
     * 方法二：in-place
     * */
    public int FindGreatestSumOfSubArray_2(int[] array) {
        if (array == null || array.length < 1) {
            return 0;
        }
        if (array.length < 2) {
            return array[0];
        }
        int ret = array[0];
        int sum = array[0];
        for (int i = 1; i < array.length; i++) {
            if (sum < 0) {
                sum = 0;
            }
            sum += array[i];
            ret = Math.max(ret, sum);
        }
        return ret;
    }

    //JZ-31.整数中1出现的次数（从1-n）
    /*
     * 一次考虑各个位置上为1的可能
     * ret:计数器
     * d=1,10,100,1000...代表个,十,百,千为一的情况下1后面位的取值可能
     * k=10*d...用来分割数字(分割为t-r)
     * t=n/k...代表1前面位取值的可能(0-t-1)
     * 考虑如果r>=d,代表1位置上原本的数字大于1,1前面可以取t(1种可能)1后面可以取min(r-d+1,d)种可能
     *      解释一下min(r-d+1,d):
     *          如果r的最高位=1,比如r=188,d=100,那么1后面共有00-88共r-d+1=89种可能
     *          如果r的最高位>2,比如r=299,d=100,那么1后面共有00-99共d种可能,因为1位置不能变.
     * */
    public int NumberOf1Between1AndN_Solution(int n) {
        int ret = 0;
        int k = 1;
        int t = n / k;
        int r = 0;//n - k * t;
        int d = 1;
        do {
            k *= 10;
            t = n / k;
            r = n - k * t;
            d = k / 10;
            ret += t * d;
            if (r >= d) {
                ret += Math.min(d, r - d + 1);
            }
        } while (t != 0);
        return ret;
    }

    //JZ-32.把数组排成最小的数
    /*
     * 总体原则:小的放前面 大的放后面.比如1 2 3->123
     * 对于两个数字a,b,在最后组成的这个数里一定是a在b前或者b在a前.
     * 即...a...b...或者...b...a...;...处对应的数字相同,只需分辨小的即可.
     * 等价于比较a[0][a1]...b[0]b[1]...和b[0]b[1]...a[0]a[1]...
     * 也就是比较a+b和b+a(字符串拼接)
     * */
    public String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length < 1) {
            return "0";
        }
        Integer[] integers = new Integer[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            integers[i] = numbers[i];
        }
        Arrays.sort(integers,
                (a, b) -> (Integer.parseInt(a.toString() + b.toString()))
                        - (Integer.parseInt(b.toString() + a.toString())));
        String ret = "";
        for (Integer item : integers) {
            ret += item;
        }
        return ret;
    }

    //JZ-33.丑数
    /*
     * 丑数是以2、3、5为因子组成的
     * 新的丑数可以由已经有的丑数×2、×3、×5得来
     * 按照递增规则，每次应该生成最小的新的丑数
     * 最小的新的丑数只有三种可能分别是×2、3、5得来
     * 令p2、p3、p5指向已经有的丑数，他们分别代表×2、3、5能得到对应最小丑数的已经有的丑数
     * 下一个丑数就是2xp2、3xp3、5xp5里面最小的。
     * 然后讨论更新p2、p3、p5的规则：
     * 对于新的丑数而言，假如他能被2整除，那么它一定是通过2xp2得来；如果不是，就与p的定义矛盾
     *   对于这种情况显然2xp2已经不是最小新丑数，需要更新p2
     *   更新操作：p2=p2的下一个丑数。
     * */
    public int GetUglyNumber_Solution(int index) {
        if (index == 0) {
            return 0;
        }

        int[] ret = new int[index];
        int p2 = 0, p3 = 0, p5 = 0;

        ret[0] = 1;
        for (int i = 1; i < index; i++) {
            int newUgly = Math.min(2 * ret[p2], 3 * ret[p3]);
            newUgly = Math.min(5 * ret[p5], newUgly);
            if (newUgly % 2 == 0) {
                p2++;
            }
            if (newUgly % 3 == 0) {
                p3++;
            }
            if (newUgly % 5 == 0) {
                p5++;
            }
            ret[i] = newUgly;
        }
        return ret[index - 1];
    }

    //JZ-34.第一个只出现一次的字符
    /*
     * 挨个读取字符，入队，用map存储每个字符的出现次数
     * 如果读到队头超过1次的，循环出队
     * 每出队一次，idx++
     *     * */
    public int FirstNotRepeatingChar(String str) {
        if (str == null || str.length() < 1) {
            return -1;
        }

        Queue<Character> queue = new LinkedList<>();
        Map<Character, Integer> map = new HashMap<>();
        int idx = 0;
        int n = str.length();
        for (int i = 0; i < n; i++) {
            char ch = str.charAt(i);
            int times = 1;
            if (map.containsKey(ch)) {
                times += map.get(ch);
            }
            map.put(ch, times);
            queue.offer(ch);

            while (!queue.isEmpty() && map.get(queue.peek()) > 1) {
                idx++;
                queue.poll();
            }
        }

        if (queue.isEmpty()) {
            return -1;
        }
        return idx;
    }

    //JZ-35.数组中的逆序对
    /*
     * 分治。
     * start-mid-end
     * f(arr,start,end):arr[start,end]中的逆序对。顺便对arr(start,end)排序
     * f(arr,start,end)=f(arr,start,mid)+f(arr,mid+1,end)+左-右逆序对数
     * 左-右逆序对数计算：对于左边的每一个数，计算其对应的逆序对，求和，复杂度O(n)
     * T(n)=2T(n/2)+n.T(n)=nlogn.
     * */
    public int InversePairs(int[] array) {
        if (array == null || array.length < 1) {
            return 0;
        }
        return f(array, 0, array.length - 1) % 1000000007;
    }

    public int f(int[] array, int start, int end) {
        if (start == end) {
            return 0;
        }
        int ret = 0;
        int mid = (start + end) / 2;
        ret += f(array, start, mid) % 1000000007;
        ret = ret % 1000000007;
        ret += f(array, mid + 1, end) % 1000000007;
        ret = ret % 1000000007;
        int i = start, j = mid + 1;
        while (i <= mid) {
            while (j <= end && array[j] < array[i]) {
                j++;
            }
            ret += (j - mid - 1) % 1000000007;
            ret = ret % 1000000007;
            i++;
        }
        //排序
        int[] sortedArr = new int[end - start + 1];
        i = start;
        j = mid + 1;
        int k = 0;
        while (i <= mid && j <= end) {
            if (array[i] < array[j]) {
                sortedArr[k++] = array[i++];
            } else {
                sortedArr[k++] = array[j++];
            }
        }
        while (i <= mid) {
            sortedArr[k++] = array[i++];
        }
        while (j <= end) {
            sortedArr[k++] = array[j++];
        }
        for (i = 0; i < end - start + 1; i++) {
            array[start + i] = sortedArr[i];
        }
        return ret;
    }

    //JZ-36.两个链表的第一个公共节点
    /*
     * 先算出两个链表的长度，然后让长的后移，直到相等
     * 然后同时后移，直到公共节点
     * */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }

        int len1 = 0, len2 = 0;
        ListNode list1 = pHead1, list2 = pHead2;

        while (list1 != null) {
            len1++;
            list1 = list1.next;
        }
        while (list2 != null) {
            len2++;
            list2 = list2.next;
        }
        list1 = pHead1;
        list2 = pHead2;
        while (len1 < len2) {
            list2 = list2.next;
            len2--;
        }
        while (len1 > len2) {
            list1 = list1.next;
            len1--;
        }
        while (list1 != list2) {
            list1 = list1.next;
            list2 = list2.next;
        }
        return list1;
    }

    //JZ-37.数字在排序数组中出现的次数
    /*
     * 先找到数字所在位置，然后左右搜索
     * */
    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length < 1) {
            return 0;
        }

        int idx = Arrays.binarySearch(array, k);
        if (idx < 0) {
            return 0;
        }
        int cnt = 0;
        for (int i = idx; i >= 0; i--) {
            if (array[i] != array[idx]) {
                break;
            }
            cnt++;
        }
        for (int i = idx + 1; i < array.length; i++) {
            if (array[i] != array[idx]) {
                break;
            }
            cnt++;
        }
        return cnt;
    }

    //JZ-38.二叉树的深度
    /*
     * 递归：略
     * 层次遍历
     * */
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        int height = 0;

        queue.offer(root);

        while (!queue.isEmpty()) {
            height++;
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return height;
    }

    //JZ-39.平衡二叉树
    /*
     * 递归：logn
     * root是平衡二叉树等价于root.left是root.right是且高度差小于2
     * */
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        int[] height = new int[1];
        return IsBalanced_Solution(root, height);
    }

    public boolean IsBalanced_Solution(TreeNode root, int[] height) {
        int left, right;
        if (root == null) {
            height[0] = 0;
            return true;
        }
        if (!IsBalanced_Solution(root.left, height)) {
            return false;
        }
        left = height[0];
        if (!IsBalanced_Solution(root.right, height)) {
            return false;
        }
        right = height[0];
        if (left - right < -1 || left - right > 1) {
            return false;
        }
        height[0] = 1 + Math.max(left, right);
        return true;
    }

    //JZ-40.数组中只出现一次的数字
    /*
     * 设这两个数字为x，y。
     * 将所有数字异或，最后结果就是x^y
     * x和y不等，于是必有一位不同，按照这个不同位分为两个部分。
     * 分别执行异或，即可得到x、y
     * */
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        int t = 0;
        for (int item : array) {
            t ^= item;
        }
        int flag = 1;
        while ((t & flag) == 0) {
            flag = flag << 1;
        }
        int x = 0, y = 0;
        for (int item : array) {
            if ((flag & item) == 0) {
                x ^= item;
            } else {
                y ^= item;
            }
        }
        num1[0] = x;
        num2[0] = y;
    }

    //JZ-41.和为S的连续正数序列
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (sum < 3) {
            return ret;
        }

        int start = 1, end = 2, lim = sum / 2;
        sum -= 3;
        while (start <= lim) {
            if (sum == 0) {
                ArrayList<Integer> arrayList = new ArrayList<>();
                for (int i = start; i <= end; i++) {
                    arrayList.add(i);
                }
                ret.add(arrayList);
            }
            if (sum < 0) {
                sum += start++;
            } else {
                sum -= ++end;
            }
        }
        return ret;
    }

    //JZ-42.和为S的两个数字
    //根据均值不等式，和相同的两个数相差越大，积越小
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        int i = 0, j = array.length - 1;
        ArrayList<Integer> ret = new ArrayList<>();

        if (array == null || array.length < 1) {
            return ret;
        }

        sum -= array[i];
        sum -= array[j];
        while (i < j) {
            if (sum == 0) {
                ret.add(array[i]);
                ret.add(array[j]);
                break;
            } else if (sum < 0) {
                sum += array[j] - array[j - 1];
                j--;
            } else {
                sum += array[i] - array[i + 1];
                i++;
            }
        }
        return ret;
    }

    //JZ-43.左旋转字符串
    public String LeftRotateString(String str, int n) {
        if (str == null || str.length() < 2 || n % str.length() == 0) {
            return str;
        }
        n = n % str.length();
        return str.substring(n) + str.substring(0, n);
    }

    //JZ-44.反转单词顺序列
    public String ReverseSentence(String str) {
        if (str == null || str.matches(" *")) {
            return str;
        }
        String[] strings = str.split(" ");
        String ret = null;
        for (String string : strings) {
            if (ret == null) {
                ret = string;
            } else {
                ret = string + " " + ret;
            }
        }
        return ret;
    }

    //JZ-45.扑克牌顺子
    public boolean isContinuous(int[] numbers) {
        Set<Integer> set = new HashSet<>();
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE, cnt = 0;
        for (int item : numbers) {
            if (item == 0) {
                cnt++;
            } else {
                min = Math.min(min, item);
                max = Math.max(max, item);
                set.add(item);
            }
        }
        return cnt + set.size() == numbers.length && max - min + 1 <= numbers.length;
    }

    //JZ-46.孩子们的游戏
    /*
     * f(n,m):表示n个人报数剩下的人
     * f(n,m)=(f(n-1,m)+m)%n
     * f(n,m)去掉编号为(m-1)%n的，于是在剩下的n-1个中，从m开始并且数f(n-1,m)为最终结果
     * */
    public int LastRemaining_Solution(int n, int m) {
        if (n < 1) {
            return -1;
        }
        if (n == 1) {
            return 0;
        }
        return (LastRemaining_Solution(n - 1, m) + m) % n;
    }

    //JZ-47.求1+2+...n
    /*
     * f(n)=f(n-1)+n;
     * f(1)=1;f(0)=0;
     * 条件语句借助&&实现（如果前面为false不会执行后面的）
     * */
    public int Sum_Solution(int n) {
        //n(n-1)/2
        int[] ret = new int[1];
        Sum_Solution(n, ret);
        return ret[0];
    }

    public boolean Sum_Solution(int n, int ret[]) {
        //n(n-1)/2
        ret[0] += n;
        return n == 1 || Sum_Solution(n - 1, ret);
    }

    //JZ-48.不用加减乘除做加法
    public int Add(int num1, int num2) {
        int c = 0;
        int ret = 0;
        for (int i = 0; i < 32; i++) {
            int t = 1 << i;
            int a = t & num1, b = t & num2;
            /*
             * if a[i]!=b[i]:ret[i]=!c
             * else ret[i]=c
             *
             * if a[i]!=b[i]:c=c
             * else if a[i]==b[i]==0:c=0;
             * else c=1
             * */
            if (a != b) {
                ret |= (c << i) ^ t;//!c
                //c=c;
            } else {
                ret |= c << i;
                if (a != 0) {
                    c = 1;
                } else {
                    c = 0;
                }
            }
        }
        return ret;
    }

    //JZ-49.吧字符串转换成整数
    public int StrToInt(String str) {
        if (str == null || str.length() < 1) {
            return 0;
        }

        char[] arr = str.toCharArray();
        int ret = 0;
        boolean neg = false;

        switch (arr[0]) {
            case '+':
                break;
            case '-':
                neg = true;
                break;
            default:
                if (Character.isDigit(arr[0])) {
                    ret = arr[0] - '0';
                } else {
                    return 0;
                }
                break;
        }

        for (int i = 1; i < arr.length; i++) {
            if (!Character.isDigit(arr[i])) {
                ret = 0;
                break;
            }
            ret = ret * 10 + arr[i] - '0';
        }
        if (neg) {
            ret = -ret;
        }
        return ret;
    }

    //JZ-50.数组中重复的数字
    /*
     * 解法一：Set
     * 挨个放入，放入之前检测是否存在。
     * 空间：O(n)，时间O(n)
     * 解法二：inplace
     * 将每一个数字放入对应位置
     * 如果当前数字已经在对应位置，那么下一个
     * 如果不在，那么和对应位置所在数交换，并且试图放置换来的数字（如果两者相等，那么返回）
     * 空间：O(1)，时间O（n）
     * */
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        int i = 0;
        while (i < length) {
            int k = numbers[i];
            if (k == i) {
                i++;
            } else if (numbers[k] == k) {
                duplication[0] = k;
                return true;
            } else {
                numbers[i] = numbers[k];
                numbers[k] = k;
            }
        }
        return false;
    }

    //JZ-51.构建乘积数组
    public int[] multiply(int[] A) {
        if (A == null || A.length < 1) {
            return A;
        }
        int ret[] = new int[A.length];
        Arrays.fill(ret, 1);
        int left = 1, right = 1;
        for (int i = 0; i < ret.length; i++) {
            ret[i] *= left;
            left *= A[i];
        }
        for (int i = ret.length - 1; i >= 0; i--) {
            ret[i] *= right;
            right *= A[i];
        }
        return ret;
    }
}
























