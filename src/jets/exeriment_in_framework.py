import os
import sys
import pytest
from PIL import Image
import torch
from training.main import main
	
	
main([
	'--save-frequency', '1',
	'--zeroshot-frequency', '1',
	'--dataset-type', "synthetic",
	'--train-num-samples', '16',
	'--warmup', '1',
	'--batch-size', '4',
	'--lr', '1e-3',
	'--wd', '0.1',
	'--epochs', '1',
	'--workers', '2',
	'--model', 'RN50'
	])