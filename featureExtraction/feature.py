#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import re
from featureExtraction.descproteins import *

from featureExtraction.pubscripts import read_fasta_sequences, save_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="Generating various numerical representation schemes for protein sequences")
    parser.add_argument("--file", required=True, help="input fasta file")
    parser.add_argument("--method", required=True,
                        choices=['AAC', 'DPC', 'GTPC', 'CTriad', 'SOCNumber', 'QSOrder',
                                 ],
                        help="the encoding type")
    parser.add_argument("--path", dest='filePath',
                        help="data file path used for 'PSSM', 'SSEB(C)', 'Disorder(BC)', 'ASA' and 'TA' encodings")
    parser.add_argument("--order", dest='order',
                        choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'],
                        help="output order for of Amino Acid Composition (i.e. AAC, DPC) descriptors")
    parser.add_argument("--userDefinedOrder", dest='userDefinedOrder',
                        help="user defined output order for of Amino Acid Composition (i.e. AAC, DPC) descriptors")
    parser.add_argument("--format", choices=['csv', 'tsv', 'svm', 'weka', 'tsv_1'], default='svm',
                        help="the encoding type")
    parser.add_argument("--out", help="the generated descriptor file")
    args = parser.parse_args()
    print(args)
    fastas = read_fasta_sequences.read_protein_sequences(args.file)
    userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }
    myOrder = myAAorder[args.order] if args.order != None else 'ACDEFGHIKLMNPQRSTVWY'
    kw = {'path': args.filePath, 'order': myOrder, 'type': 'Protein'}
    cmd = args.method + '.' + args.method + '(fastas, **kw)'
    print('Descriptor type: ' + args.method)
    encodings = eval(cmd)
    out_file = args.out if args.out != None else 'encoding.txt'
    save_file.save_file(encodings, args.format, out_file)
