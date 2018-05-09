#!/bin/env python
#-*- encoding: utf-8 -*-

from math import log, sqrt
import copy
import codecs
import utils


class DecisionTree:

    def __init__(self, feature=-1, value=None, true_branch=None, 
                 false_branch=None, results={}, result=None, error=0):
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        # Classification result at current node (majority class).
        # These three variables will change when building, evaluating or pruning a tree.
        self.result = result
        self.results = results
        self.error = error

    @classmethod
    def _divide_set(cls, dataset, feature, value):
        if isinstance(value, int) or isinstance(value, float):
            func = lambda data: data[feature] >= value
        else:
            func = lambda data: data[feature] == value

        true_set = [data for data in dataset if func(data)]
        false_set = [data for data in dataset if not func(data)]
        return true_set, false_set

    @classmethod
    def entropy(cls, dataset):
        log2 = lambda x: log(x) / log(2)
        rst = utils.count(dataset)
        entropy = 0.0

        for r in rst:
            p = float(rst[r]) / len(dataset)
            entropy -= p * log2(p)
        return entropy

    @classmethod
    def gini(cls, dataset):
        rst = utils.count(dataset)
        gini = 1.0

        for r in rst:
            gini -= (rst[r] / len(dataset)) ** 2
        return gini

    @classmethod
    def build_tree(cls, dataset, func):
        if len(dataset) == 0:
            return DecisionTree()

        best_gain = 0.0
        best_feature = None
        best_split = None
        cur_score = func(dataset)
        feature_cnt = len(dataset[0]) - 1

        results = utils.count(dataset)
        result = sorted(results.items(), key=lambda x: x[1], reverse=True)[0][0]
        error = 0
        for k, v in results.items():
            if k != result:
                error += v

        # Choose the best feature
        for i in range(feature_cnt):

            unique_values = list(set([data[i] for data in dataset]))
            for v in unique_values:
                true_set, false_set = cls._divide_set(dataset, i, v)

                p_true = float(len(true_set)) / len(dataset)
                p_false = 1 - p_true
                gain = cur_score - p_true * \
                    func(true_set) - p_false * func(false_set)

                if gain > best_gain and len(true_set) and len(false_set):
                    best_gain = gain
                    best_feature = (i, v)
                    best_split = (true_set, false_set)

        if not best_gain:
            return DecisionTree(result=result, results=results, error=error)

        true_branch = cls.build_tree(best_split[0], func)
        false_branch = cls.build_tree(best_split[1], func)
        return DecisionTree(feature=best_feature[0], value=best_feature[1], \
                    true_branch=true_branch, false_branch=false_branch, \
                    result=result, results=results, error=error)

    @classmethod
    def plot_tree(cls, tree, headings, filepath=None):

        def _tree_to_str(tree, indent='\t\t'):
            # General output
            output = str(tree.result) + ' ' + str(tree.results) + \
                    ' err=' + str(tree.error)

            # Leaf node
            if not (tree.true_branch or tree.false_branch):
                return output

            if tree.feature in headings:
                col = headings[tree.feature]

            if isinstance(tree.value, int) or isinstance(tree.value, float):
                decision = ' %s >= %s ?' % (col, tree.value)
            else:
                decision = ' %s == %s ?' % (col, tree.value)

            true_branch = indent + 'yes -> ' + \
                _tree_to_str(tree.true_branch, indent + '\t\t')
            false_branch = indent + 'no  -> ' + \
                _tree_to_str(tree.false_branch, indent + '\t\t')
            return output + decision + '\n' + true_branch + '\n' + false_branch

        str_tree = _tree_to_str(tree)

        if filepath:
            with codecs.open(filepath, 'w', encoding='utf-8') as f:
                f.write(str_tree)
        else:
            print(str_tree)

    @classmethod
    def evaluate(cls, tree, dataset):

        def _evaluate(eval_tree, dataset):
            eval_tree.results = utils.count(dataset)
            eval_tree.error = 0
            for k, v in eval_tree.results.items():
                if k != eval_tree.result:
                    eval_tree.error += v

            # Leaf node
            if not (eval_tree.true_branch or eval_tree.false_branch):
                return eval_tree.error

            true_set = []
            false_set = []
            for data in dataset:
                v = data[eval_tree.feature]
                if isinstance(v, int) or isinstance(v, float):
                    if v >= eval_tree.value:
                        true_set.append(data)
                    else:
                        false_set.append(data)
                else:
                    if v == eval_tree.value:
                        true_set.append(data)
                    else:
                        false_set.append(data)
            return cls.evaluate(eval_tree.true_branch, true_set) + \
                    cls.evaluate(eval_tree.false_branch, false_set)

        # Deepcopy the tree to store test set info
        eval_tree = copy.deepcopy(tree)

        return _evaluate(eval_tree, dataset)

    @classmethod
    def count_leaves(cls, tree):
        if not (tree.true_branch or tree.false_branch):
            return 1
        return cls.count_leaves(tree.true_branch) + cls.count_leaves(tree.false_branch)

    @classmethod
    def reduced_error_pruning(cls, tree):
        # Bottom-up, left-to-right
        # Leaf node
        if not (tree.true_branch or tree.false_branch):
            return tree.error
        error_true = cls.reduced_error_pruning(tree.true_branch)
        error_false = cls.reduced_error_pruning(tree.false_branch)

        # Prune its subtree if it has less error
        if tree.error <= error_true + error_false:
            tree.true_branch = None
            tree.false_branch = None
            return tree.error
        else:
            return error_true + error_false

    @classmethod
    def _pessimistic_error(cls, tree):
        # Leaf node
        if not (tree.true_branch or tree.false_branch):
            return tree.error + 0.5
        return cls._pessimistic_error(tree.true_branch) + \
                cls._pessimistic_error(tree.false_branch)

    @classmethod
    def top_down_pessimistic_pruning(cls, tree):
        # Top-down, left-to-right
        # Leaf node
        if not (tree.true_branch or tree.false_branch):
            return

        error_leaf = tree.error + 0.5
        error_subtree = cls._pessimistic_error(tree)
        p = 1 - error_subtree / sum(tree.results.values())
        # p < 0 in few cases
        p = 0 if p < 0 else p

        if error_leaf <= error_subtree + sqrt(error_subtree * p):
            tree.true_branch = None
            tree.false_branch = None
        else:
            cls.top_down_pessimistic_pruning(tree.true_branch)
            cls.top_down_pessimistic_pruning(tree.false_branch)

    @classmethod
    def bottom_up_pessimistic_pruning(cls, tree):
        # Bottom-up, left-to-right
        sum_ = sum(tree.results.values())
        p = (1 + tree.error) / (2 + sum_)
        # Laplace estimate
        error_leaf = sum_ * (p + 1.15 * sqrt(p * (1 - p) / (sum_ + 2)))

        # Leaf node
        if not (tree.true_branch or tree.false_branch):
            return error_leaf

        error_subtree = cls.bottom_up_pessimistic_pruning(tree.true_branch) + \
                        cls.bottom_up_pessimistic_pruning(tree.false_branch)
        if error_leaf <= error_subtree:
            tree.true_branch = None
            tree.false_branch = None
            return error_leaf
        return error_subtree

    @classmethod
    def minimum_error_pruning(cls, tree):
        # Bottom-up, left-to-right
        sum_ = sum(tree.results.values())
        # (n(error) + k - 1) / n(all) + k
        error_leaf = (tree.error + 2) / (sum_ + 3)

        # Leaf node
        if not (tree.true_branch or tree.false_branch):
            return sum_, error_leaf

        sum_true, error_true = cls.minimum_error_pruning(tree.true_branch)
        sum_false, error_false = cls.minimum_error_pruning(tree.false_branch)
        error_subtree = sum_true / sum_ * error_true + sum_false / sum_ * error_false

        if error_leaf <= error_subtree:
            tree.true_branch = None
            tree.false_branch = None
            return sum_, error_leaf
        return sum_, error_subtree
