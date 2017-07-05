#!/bin/bash

python sort_of_clevr_generator.py

python main.py --model=RN      --epochs=20

python main.py --model=CNN_MLP --epochs=100
