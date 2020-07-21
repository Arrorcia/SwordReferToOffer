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

}
