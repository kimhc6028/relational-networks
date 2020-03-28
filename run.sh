#!/bin/bash

python sort_of_clevr_generator.py

python main.py --model=RN      --epochs=20      --relation-type=binary

python main.py --model=RN      --epochs=20      --relation-type=ternary

python main.py --model=CNN_MLP --epochs=100
