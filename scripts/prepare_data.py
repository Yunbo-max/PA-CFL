#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Prepare data for all experiments.
# Downloads/loads the DataCo Supply Chain Dataset and produces
# a unified processed pickle file used by all methods.
#
# Usage: python scripts/prepare_data.py --raw_data data/DataCoSupplyChainDataset.csv

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import load_and_prepare_data


def main():
    parser = argparse.ArgumentParser(description='Prepare data for PA-CFL experiments')
    parser.add_argument('--raw_data', type=str,
                        default='data/DataCoSupplyChainDataset.csv',
                        help='Path to raw DataCo Supply Chain CSV')
    parser.add_argument('--output', type=str,
                        default='data/processed/datasets.pkl',
                        help='Output path for processed datasets')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.raw_data):
        print(f"Error: Raw data not found at {args.raw_data}")
        print("Please download the DataCo Supply Chain Dataset:")
        print("  https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")
        print(f"  and place DataCoSupplyChainDataset.csv in data/")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Processing DataCo Supply Chain Dataset...")
    datasets = load_and_prepare_data(
        args.raw_data, args.output,
        test_size=args.test_size, random_state=args.seed)

    print(f"\nProcessed {len(datasets)} regions:")
    for region, data in datasets.items():
        n_train = data['train_features'].shape[0]
        n_test = data['test_features'].shape[0]
        n_feat = data['num_features']
        print(f"  {region}: {n_train} train / {n_test} test samples, {n_feat} features")

    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
