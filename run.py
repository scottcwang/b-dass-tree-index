#!/usr/bin/env python3

import argparse
import time
import itertools

import numpy
import sortedcontainers

import btree


def generate_workload(
    data,
    insert_params,
    lookup_params,
    insertion_lookup_interspersion
):
    while True:
        if (
            numpy.random.default_rng().random()
            < insertion_lookup_interspersion
        ):
            # Insert
            chosen_t = numpy.random.default_rng().choice(
                len(insert_params),
                p=[t['p'] for t in insert_params]
            )
            k = numpy.random.default_rng().beta(
                insert_params[chosen_t]['a'],
                insert_params[chosen_t]['b']
            )
            data[k] = {
                'v': numpy.random.default_rng().bytes(1),
                'time': []
            }
        else:
            # Lookup
            if len(data) == 0:
                continue
            chosen_t = numpy.random.default_rng().choice(
                len(lookup_params),
                p=[t['p'] for t in lookup_params]
            )
            k = data.peekitem(
                min(
                    data.bisect(
                        numpy.random.default_rng().beta(
                            lookup_params[chosen_t]['a'],
                            lookup_params[chosen_t]['b']
                        )
                    ), len(data) - 1
                )
            )[0]
        yield k


def harness_test(n, btree_args, experiment_args):
    t = btree.Tree(**btree_args)
    data = sortedcontainers.SortedDict()

    for k in itertools.islice(
        generate_workload(data, **experiment_args),
        n
    ):
        start = time.perf_counter()
        if len(data[k]['time']) == 0:
            t.insert(k, data[k]['v'])
        else:
            assert t.search(k) == data[k]['v']
        end = time.perf_counter()
        data[k]['time'].append(end - start)
        assert t.is_consistent()

    return data


if __name__ == '__main__':
    a = argparse.ArgumentParser()

    a.add_argument('out_filename', type=str)

    a.add_argument('--n', type=int)

    a.add_argument('--btree_node_capacity', type=int)
    a.add_argument('--btree_error_limit', type=int)
    a.add_argument('--btree_knots_limit', type=int)
    a.add_argument('--btree_min_knots', type=int)

    a.add_argument('--experiment_insert_params', type=str, nargs='*')
    a.add_argument('--experiment_lookup_params', type=str, nargs='*')
    a.add_argument('--experiment_insertion_lookup_interspersion', type=float)

    args = a.parse_args()

    data = harness_test(
        n=args.n,
        btree_args={
            'node_capacity': args.btree_node_capacity,
            'error_limit': args.btree_error_limit,
            'knots_limit': args.btree_knots_limit,
            'min_knots': args.btree_min_knots
        },
        experiment_args={
            'insert_params': [
                {
                    'p': float(i.split(',')[0]),
                    'a': float(i.split(',')[1]),
                    'b': float(i.split(',')[2])
                }
                for i in args.experiment_insert_params
            ],
            'lookup_params': [
                {
                    'p': float(i.split(',')[0]),
                    'a': float(i.split(',')[1]),
                    'b': float(i.split(',')[2])
                }
                for i in args.experiment_lookup_params
            ],
            'insertion_lookup_interspersion':
            args.experiment_insertion_lookup_interspersion
        }
    )

    with open(args.out_filename, 'w') as f:
        f.write(
            '\n'.join(
                ','.join(
                    map(
                        str,
                        [k] + v['time']
                    )
                )
                for k, v in data.items()
            )
        )
