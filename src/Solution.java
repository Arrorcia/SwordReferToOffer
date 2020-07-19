import java.util.ArrayList;
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
}
