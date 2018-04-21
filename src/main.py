#!/bin/env python
#-*- encoding: utf-8 -*-

import random
import utils
from tree import DecisionTree


def main():
    '''
    # Prepare dataset
    headings, dataset = utils.load_dataset()
    random.shuffle(dataset)

    # Experiment 1:
    train_data = dataset[:30000]
    test_data = dataset[30000:40000]
    dt = DecisionTree.build_tree(train_data, DecisionTree.entropy)
    err = DecisionTree.count_error(dt, test_data)
    print('Experiment 1, accuracy: %d/%d = %f' % \
        (len(test_data) - err, len(test_data), (len(test_data) - err) / len(test_data)))
    
    # Experiment 2:
    train_data = dataset[:30000]
    test_data = dataset[30000:40000]
    dt = DecisionTree.build_tree(train_data, DecisionTree.gini)
    err = DecisionTree.count_error(dt, test_data)
    print('Experiment 2, accuracy: %d/%d = %f' % \
        (len(test_data) - err, len(test_data), (len(test_data) - err) / len(test_data)))
    
    # Experiment 3:
    train_data = dataset[:15000]
    prune_data = dataset[15000:30000]
    test_data = dataset[30000:40000]
    dt = DecisionTree.build_tree(train_data, DecisionTree.gini)
    err_prune = DecisionTree.count_error(dt, prune_data)
    err_test = DecisionTree.count_error(dt, test_data)
    print('Experiment 3, accuracy: %d/%d = %f' % \
        (len(prune_data) - err_prune, len(prune_data), (len(prune_data) - err_prune) / len(prune_data)))
    print('Experiment 3, accuracy: %d/%d = %f' % \
        (len(test_data) - err_test, len(test_data), (len(test_data) - err_test) / len(test_data)))
    DecisionTree.plot_tree(dt, headings)
    DecisionTree.reduced_error_pruning(dt)
    DecisionTree.plot_tree(dt, headings)
    err_prune = DecisionTree.count_error(dt, prune_data)
    err_test = DecisionTree.count_error(dt, test_data)
    print('Experiment 3, accuracy: %d/%d = %f' % \
        (len(prune_data) - err_prune, len(prune_data), (len(prune_data) - err_prune) / len(prune_data)))
    print('Experiment 3, accuracy: %d/%d = %f' % \
        (len(test_data) - err_test, len(test_data), (len(test_data) - err_test) / len(test_data)))
    '''
    return

if __name__ == '__main__':
	main()
