#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 03:18:35 2024

@author: noob
"""

"""
arbml/CIDAR
FreedomIntelligence/alpaca-gpt4-arabic

X ? Wrong TRanslation , Yasbok/Alpaca_arabic_instruct
X FreedomIntelligence/Evol-Instruct-Arabic-GPT4
X FreedomIntelligence/sharegpt-arabic
X ? cause translate more , akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft
"""

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("arbml/CIDAR", split='train')

dataset = dataset.remove_columns('index')
df = pd.DataFrame({
    'instruction': dataset['instruction'],
    'output': dataset['output']
})


df = df.drop_duplicates()



dataset1 = load_dataset("FreedomIntelligence/alpaca-gpt4-arabic", split='train')

dataset1 = dataset1.remove_columns('id')


df0 = pd.DataFrame({
    'instruction': list(map(lambda conv: conv[0]['value'], dataset1['conversations'])),
    'output': list(map(lambda conv: conv[1]['value'], dataset1['conversations']))
})

df0 = df0.drop_duplicates()


Final_df = pd.concat([df, df0], ignore_index=True)

del dataset ,dataset1 ,df,df0

#Final_df.to_csv('dataset.csv', index=False)


######################################################################


