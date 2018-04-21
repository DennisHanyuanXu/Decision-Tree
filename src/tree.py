#!/bin/env python
#-*- encoding: utf-8 -*-

from math import log
import utils


class DecisionTree:

    def __init__(self, feature=-1, value=None, true_branch=None, 
                 false_branch=None, results={}, result=None, error=0):
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
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
        result = sorted(utils.count(dataset).items(), key=lambda x: x[1], \
                        reverse=True)[0][0]

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
            return DecisionTree(result=result)

        true_branch = cls.build_tree(best_split[0], func)
        false_branch = cls.build_tree(best_split[1], func)
        return DecisionTree(feature=best_feature[0], value=best_feature[1], \
                    true_branch=true_branch, false_branch=false_branch, result=result)

    @classmethod
    def plot_tree(cls, tree, headings):

        def tree_to_str(tree, indent='\t\t'):
            output = str(tree.result) + ' '
            output += str(tree.results) + ' ' if tree.results else ''
            output += str(tree.error) + ' ' if tree.error else ''

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
                tree_to_str(tree.true_branch, indent + '\t\t')
            false_branch = indent + 'no  -> ' + \
                tree_to_str(tree.false_branch, indent + '\t\t')
            return output + decision + '\n' + true_branch + '\n' + false_branch

        print(tree_to_str(tree))


# Test
if __name__ == '__main__':
    headings, dataset = utils.load_dataset()
    t = DecisionTree.build_tree(dataset[:100], DecisionTree.gini)
    DecisionTree.plot_tree(t, headings)
