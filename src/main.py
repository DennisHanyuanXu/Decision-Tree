#!/bin/env python
#-*- encoding: utf-8 -*-

import random
import utils
import conf
from tree import DecisionTree


def main():
    # Steps to build and prune a decision tree:

    # 1. Prepare dataset.
    headings, dataset = utils.load_dataset()
    random.shuffle(dataset)
    # Split the dataset into training data, test data and pruning data if needed.
    train_data = dataset[:32000]
    test_data = dataset[32000:40000]
    # prune_data = dataset[:]


    # 2. Grow a decision tree from training data based on entropy or gini.
    dt = DecisionTree.build_tree(train_data, DecisionTree.entropy)
    # dt = DecisionTree.build_tree(train_data, DecisionTree.gini)


    # 3. Visualize the tree.
    DecisionTree.plot_tree(dt, headings, conf.org_tree_filepath)
    leaves = DecisionTree.count_leaves(dt)
    print('Leaves count before pruning: %d' % leaves)


    # 4. Run the test data through the tree.
    err = DecisionTree.evaluate(dt, test_data)
    print('Accuracy before pruning: %d/%d = %f' % \
        (len(test_data) - err, len(test_data), (len(test_data) - err) / len(test_data)))


    # 5. Prune the tree.
    #   5.1 REP: REP requires another dataset for pruning, so we need to split the dataset in a different way.

    #   5.2 PP: top-down
    DecisionTree.top_down_pessimistic_pruning(dt)

    #   5.3 PP: bottom-up.
    # DecisionTree.bottom_up_pessimistic_pruning(dt)
    
    #   5.4 MEP
    # DecisionTree.minimum_error_pruning(dt)


    # 6. Visualize the pruned tree.
    DecisionTree.plot_tree(dt, headings, conf.prn_tree_filepath)
    leaves = DecisionTree.count_leaves(dt)
    print('Leaves count after pruning: %d' % leaves)


    # 7. Check if the classification ability is improved after pruning.
    err = DecisionTree.evaluate(dt, test_data)
    print('Accuracy after pruning: %d/%d = %f' % \
        (len(test_data) - err, len(test_data), (len(test_data) - err) / len(test_data)))


if __name__ == '__main__':
	main()
