#!/usr/bin/env python
# _*_ coding: utf-8 _*_


protein_cmd_coding = {
    'AAC': ['AAC.AAC(training_data, **kw)', 'AAC.AAC(testing_data, **kw)'],
    'DPC': ['DPC.DPC(training_data, **kw)', 'DPC.DPC(testing_data, **kw)'],
    'GTPC': ['GTPC.GTPC(training_data, **kw)', 'GTPC.GTPC(testing_data, **kw)'],
    'CTriad': ['CTriad.CTriad(training_data, gap=0, **kw)', 'CTriad.CTriad(testing_data, gap=0, **kw)'],
    'SOCNumber': ['SOCNumber.SOCNumber(training_data, nlag=%d, **kw)' % int(parameters['Lag_Value']), 'SOCNumber.SOCNumber(testing_data, nlag=%d, **kw)' % int(parameters['Lag_Value'])],
    'QSOrder': ['QSOrder.QSOrder(training_data, nlag=%d, w=%f, **kw)' % (int(parameters['Lag_Value']), float(parameters['Weight_Value'])), 'QSOrder.QSOrder(testing_data, nlag=%d, w=%f, **kw)' % (int(parameters['Lag_Value']), float(parameters['Weight_Value']))],

}