#!/usr/bin/env python3

import warnings

import numpy as np
import scipy.interpolate as spi


class TreeNode:
    def __init__(self, capacity, error_limit, knots_limit, min_knots):
        self.children = np.empty(
            capacity,
            dtype=[('key', np.float_), ('value', np.object_)]
        )
        self.error_limit = error_limit
        self.knots_limit = knots_limit
        self.min_knots = min_knots
        self.knots = self.min_knots
        self.lower_bound = 0.0
        self.min_occupied_index = self.children.size - 1
        self.max_occupied_index = 0

    def guess_index(self, key):
        """
        Returns a guess for the index of key, rounded to the nearest integer.
        """
        knot_y = [
            self.min_occupied_index + (
                self.max_occupied_index - self.min_occupied_index
            ) // self.knots * i for i in range(self.knots + 1)
        ]
        knot_x = self.children[knot_y]['key']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_guess = (
                spi.interp1d(
                    knot_x, knot_y, fill_value='extrapolate', assume_sorted=True
                )
            )(key)
        if not np.isfinite(raw_guess):
            return (
                self.min_occupied_index + self.max_occupied_index
            ) // 2
        else:
            return np.around(
                np.clip(
                    raw_guess, self.min_occupied_index, self.max_occupied_index
                )
            ).astype(np.int_)

    def exponential_search(self, guess_index, key):
        """
        Returns the index above which the given key would be inserted
        (n+1 possible return values).
        """
        left = guess_index + \
            (-1 if self.children[guess_index]['key']
             > key and guess_index > 0 else 0)
        right = guess_index + \
            (1 if self.children[guess_index]['key'] <
             key and guess_index < self.children.size - 1 else 0)

        while (
            right < self.children.size - 1
            and self.children[right]['key'] < key
        ):
            left, right = right, min(
                self.children.size - 1,
                right + (right - left) * 2
            )

        while left > 0 and self.children[left]['key'] >= key:
            left, right = min(
                0,
                left - (right - left) * 2
            ), left

        return self.binary_search(left, right, key)

    def binary_search(self, left, right, key):
        """
        Returns the index above which the given key would be inserted
        (n+1 possible return values).
        """
        if left == right:
            if self.children[left]['key'] < key:
                return None, left + 1
            else:
                return self.children[left]['value'], left

        middle = left + (right - left) // 2

        if self.children[middle]['key'] < key:
            return self.binary_search(middle + 1, right, key)
        else:
            return self.binary_search(left, middle, key)

    def search(self, key):
        """
        Returns the value of this key in the map in the subtree rooted at this
        node, or None if it doesn't exist; returns the newly created node if
        this node is split.
        """
        guess_index = self.guess_index(key)
        value, correct_index = self.exponential_search(
            guess_index, key)
        while True:
            if type(value) == TreeNode:
                value, right_child = value.search(key)
                if right_child is not None:
                    right_self = self.insert(
                        right_child.children[-1]['key'], right_child
                    )
                    return value, right_self

            if value is not None:
                if (
                    correct_index is not None
                    and abs(correct_index - guess_index) > self.error_limit
                ):
                    self.knots += 1
                    if self.knots > self.knots_limit:
                        right_self = self.split()
                        return value, right_self
                return value, None

            if (
                correct_index >= self.children.size - 1
                or self.children[correct_index]['key'] > key
            ):
                return None, None

            correct_index += 1
            value = self.children[correct_index]['value']

    def update_keys_preceding_index(self, index, min_key, max_key):
        """
        Sets the key of the given index to max_key.

        If all entries in this node before the given index are vacant,
        sets all those entries' keys to min_key.

        Otherwise, sets the keys of all vacant entries before index with a
        linear interpolation between that slot's key and min_key.
        """
        self.children[index]['key'] = max_key

        if index == 0:
            return
        preceding_occupied_index = index - 1
        ct_vacant_index = 0
        while (
            self.children[preceding_occupied_index]['value'] is None
                and preceding_occupied_index >= 0
        ):
            preceding_occupied_index -= 1
            ct_vacant_index += 1
        for m, i in enumerate(range(preceding_occupied_index + 1, index)):
            self.children[i]['key'] = (
                self.children[preceding_occupied_index]['key'] + (
                    min_key - self.children[preceding_occupied_index]['key']
                ) / (ct_vacant_index + 1) * (m + 1)
                if preceding_occupied_index >= 0
                else min_key
            )

    def split(self):
        """
        Splits this node by evenly distributing the occupied entries across
        this node and a newly created node, which is returned.
        """
        occupied_children = [
            (child['key'], child['value'])
            for child in self.children
            if child['value'] is not None
        ]
        if len(occupied_children) <= 4:
            return None

        right_occupied_children = occupied_children[len(
            occupied_children) // 2:]
        right_self = TreeNode(
            self.children.size, self.error_limit, self.knots_limit, self.min_knots
        )
        for i, (key, value) in enumerate(right_occupied_children):
            insert_index = (
                i * right_self.children.size // len(right_occupied_children)
            )
            right_self.children[insert_index]['value'] = value
            right_self.update_keys_preceding_index(
                insert_index,
                (
                    value.lower_bound
                    if type(value) == TreeNode
                    else key
                ),
                key
            )
            right_self.min_occupied_index = min(
                right_self.min_occupied_index,
                insert_index
            )
            right_self.max_occupied_index = max(
                right_self.max_occupied_index,
                insert_index
            )
        for i in range(right_self.max_occupied_index, right_self.children.size):
            right_self.children[i]['key'] = right_self.children[
                right_self.max_occupied_index
            ]['key']
        right_self.lower_bound = (
            right_self.children[0]['value'].lower_bound
            if type(right_self.children[0]['value']) == TreeNode
            else right_self.children[0]['key']
        )

        self_occupied_children = occupied_children[:len(
            occupied_children) // 2]
        self.min_occupied_index = self.children.size - 1
        self.max_occupied_index = 0

        for i, (key, value) in enumerate(self_occupied_children):
            insert_index = (
                i * self.children.size // len(self_occupied_children)
            )
            self.children[insert_index]['value'] = value
            self.update_keys_preceding_index(
                insert_index,
                (
                    value.lower_bound
                    if type(value) == TreeNode
                    else key
                ),
                key
            )
            self.min_occupied_index = min(
                self.min_occupied_index,
                insert_index
            )
            self.max_occupied_index = max(
                self.max_occupied_index,
                insert_index
            )

            for j in range(
                i * self.children.size // len(self_occupied_children) + 1,
                min(
                    (i + 1) * self.children.size // len(self_occupied_children) + 1,
                    self.children.size
                )
            ):
                self.children[j]['value'] = None

        for i in range(self.max_occupied_index, self.children.size):
            self.children[i]['key'] = self.children[
                self.max_occupied_index
            ]['key']

        self.knots = 1

        return right_self

    def insert(self, key, value):
        """
        Inserts the given key-value mapping into the subtree rooted at this
        node; updates the keys that precedes the new key; returns the newly
        created node if this node is split.
        """
        self.lower_bound = min(
            self.lower_bound,
            value.lower_bound if type(value) == TreeNode else key
        )

        guess_index = self.guess_index(key)
        child_value, correct_index = self.exponential_search(
            guess_index, key)

        if type(child_value) == TreeNode:
            right_child = child_value.insert(key, value)
            if right_child is not None:
                key, value = right_child.children[-1]['key'], right_child
                self.update_keys_preceding_index(
                    correct_index,
                    child_value.lower_bound,
                    child_value.children[-1]['key']
                )
                correct_index += 1
            else:
                self.update_keys_preceding_index(
                    correct_index,
                    child_value.lower_bound,
                    child_value.children[-1]['key']
                )
                return None

        left_index = correct_index - 1
        right_index = correct_index
        while True:
            if (
                left_index >= 0
                and self.children[left_index]['value'] == None
            ):
                self.children[
                    left_index:correct_index - 1
                ] = self.children[
                    left_index + 1:correct_index
                ]
                self.children[correct_index - 1]['key'] = key
                self.children[correct_index - 1]['value'] = value
                self.update_keys_preceding_index(
                    correct_index - 1,
                    (
                        value.lower_bound
                        if type(value) == TreeNode
                        else key
                    ),
                    key
                )
                self.min_occupied_index = min(
                    self.min_occupied_index,
                    left_index
                )
                self.max_occupied_index = max(
                    self.max_occupied_index,
                    correct_index - 1
                )
                return None
            elif (
                right_index < self.children.size
                and self.children[right_index]['value'] == None
            ):
                self.children[
                    correct_index + 1:right_index + 1
                ] = self.children[
                    correct_index:right_index
                ]
                self.children[correct_index]['key'] = key
                self.children[correct_index]['value'] = value
                self.min_occupied_index = min(
                    self.min_occupied_index,
                    correct_index
                )
                self.max_occupied_index = max(
                    self.max_occupied_index,
                    right_index
                )
                return None
            else:
                left_index -= 1
                right_index += 1

            if min(
                abs(right_index - correct_index),
                abs(left_index - correct_index)
            ) >= self.error_limit:
                right_self = self.split()
                if (
                    right_self is not None
                    and self.children[-1]['key'] < key
                ):
                    right_self.insert(key, value)
                else:
                    self.insert(key, value)

                return right_self

    def is_consistent(self):
        """
        Returns whether this node is consistent - for debugging.
        """
        if not self.lower_bound <= (
            self.children[0]['value'].lower_bound
            if type(self.children[0]['value']) == TreeNode
            else self.children[0]['key']
        ):
            return False

        for i in range(self.children.size - 1):
            if (
                i < self.children.size - 1
                and self.children[i]['key'] > (
                    self.children[i+1]['value'].lower_bound
                    if type(self.children[i+1]['value']) == TreeNode
                    else self.children[i+1]['key']
                )
            ):
                return False
            if (
                type(self.children[i]['value']) == TreeNode
                and not self.children[i]['value'].is_consistent()
            ):
                return False

        return True

    def depth(self):
        """
        Returns the maximum depth of the subtree rooted at this node.
        """
        return 1 + max(
            (
                child['value'].depth()
                if type(child['value']) == TreeNode
                else 0
            )
            for child in self.children
        )

    def capacity(self):
        """
        Returns the capacity of the subtree rooted at this node.
        """
        return self.children.size + sum(
            (
                child['value'].capacity()
                if type(child['value']) == TreeNode
                else 0
            )
            for child in self.children
        )

    def count(self):
        """
        Returns the number of key-value mappings in the subtree rooted at this node.
        """
        return sum(
            (
                child['value'].count()
                if type(child['value']) == TreeNode
                else 1
                if child['value'] is not None
                else 0
            )
            for child in self.children
        )


class Tree:
    def __init__(self, node_capacity, error_limit, knots_limit, min_knots):
        self.node_capacity = node_capacity
        self.error_limit = error_limit
        self.knots_limit = knots_limit
        self.min_knots = min_knots
        self.root = TreeNode(
            self.node_capacity, self.error_limit, self.knots_limit, self.min_knots
        )

    def split_right_self(self, right_self):
        """
        Specially handles the root node if it is split.
        """
        if right_self is not None:
            new_root = TreeNode(
                self.node_capacity, self.error_limit, self.knots_limit, self.min_knots
            )

            new_root.children[
                new_root.children.size // 3
            ]['value'] = self.root
            new_root.update_keys_preceding_index(
                new_root.children.size // 3,
                self.root.lower_bound,
                self.root.children[-1]['key']
            )

            new_root.lower_bound = self.root.lower_bound
            new_root.min_occupied_index = new_root.children.size // 3

            new_root.children[
                new_root.children.size // 3 * 2
            ]['value'] = right_self
            new_root.update_keys_preceding_index(
                new_root.children.size // 3 * 2,
                right_self.lower_bound,
                right_self.children[-1]['key']
            )

            new_root.max_occupied_index = new_root.children.size // 3 * 2
            for i in range(new_root.max_occupied_index, new_root.children.size):
                new_root.children[i]['key'] = new_root.children[
                    new_root.max_occupied_index
                ]['key']

            self.root = new_root

    def insert(self, key, value):
        """
        Inserts the given key-value mapping into the tree.
        """
        right_self = self.root.insert(key, value)
        self.split_right_self(right_self)

    def search(self, key):
        """
        Returns the value of this key in the map in this tree, or None if it
        doesn't exist.
        """
        value, right_self = self.root.search(key)
        self.split_right_self(right_self)
        return value

    def is_consistent(self):
        """
        Returns whether this tree is consistent - for debugging.
        """
        return self.root.is_consistent()

    def depth(self):
        """
        Returns the maximum depth of this tree.
        """
        return self.root.depth()

    def capacity(self):
        """
        Returns the capacity of the subtree rooted at this node.
        """
        return self.root.capacity()

    def count(self):
        """
        Returns the number of key-value mappings in the subtree rooted at this node.
        """
        return self.root.count()
