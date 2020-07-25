import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

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
        return null;
    }
}
