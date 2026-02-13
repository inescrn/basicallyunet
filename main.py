import os
import sys
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import BasicallyUnet, dice_loss
from data import Data

#==========================================================

num_batch      = 10
num_workers    = 4

learning_rate     = 1e-3
min_learning_rate = 1e-5
lambda1           = lambda epochs: max(0.975 ** epochs, min_learning_rate / learning_rate)
save_frequency    = 10
nfilter           = 64
load_first        = False
beta              = 1

augment_noise = 0.025

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------

def save_model(model_file, model, optimizer, history, suffix=None):
	path = model_file if suffix is None else f'{model_file}_{suffix}.pth'
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history}, path)

#----------------------------------------------------------
def train(args):

	train_data = Data(args.data_path, noise=augment_noise)
	test_data  = Data(args.test_path)

	print('Training data:', len(train_data), 'Testing data:', len(test_data))

	num_batches  = np.ceil(len(train_data) / num_batch).astype('int')
	num_batches += np.ceil(len(test_data)  / num_batch).astype('int')

	model = BasicallyUnet(in_channels=1, base_channels=nfilter).to(DEVICE)

	train_loader = DataLoader(train_data, batch_size=num_batch, num_workers=num_workers, shuffle=True)
	test_loader  = DataLoader(test_data,  batch_size=num_batch, num_workers=num_workers)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

	history = {
		'train_loss': [], 'test_loss': [],
		'toc': [],
		'train_dice': [], 'train_mse': [],
		'test_dice':  [], 'test_mse':  []
	}

	h = len(history['train_loss'])

	if load_first and os.path.exists(args.model_file):
		checkpoint = torch.load(args.model_file, weights_only=True)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		history = checkpoint['history']
		del checkpoint

	for n in range(h, args.num_epoch):
		tic = time.time()

		# ---- Training ----
		model.train()
		train_running_loss = 0.0

		for i, (images, labels) in enumerate(train_loader):
			print('Epoch {0} of {1}, Batch {2} of {3} [training]...'.format(n+1, args.num_epoch, i+1, num_batches), end='\r', flush=True)

			labels = labels.to(DEVICE)
			images = images.to(DEVICE)

			optimizer.zero_grad(set_to_none=True)

			output = model(images)

			train_loss = dice_loss(output, labels)
			train_loss.backward()
			optimizer.step()

			train_running_loss += train_loss.item()

		before_lr = optimizer.param_groups[0]["lr"]
		scheduler.step()
		after_lr = optimizer.param_groups[0]["lr"]
		print("\nEpoch {}: lr {:.2e} -> {:.2e}".format(n+1, before_lr, after_lr))

		# ---- Validation ----
		model.eval()
		test_running_loss = 0.0

		with torch.no_grad():
			for j, (images, labels) in enumerate(test_loader):
				print('Epoch {0} of {1}, Batch {2} of {3} [validation]...'.format(n+1, args.num_epoch, j+1, len(test_loader)), end='\r', flush=True)

				labels = labels.to(DEVICE)
				images = images.to(DEVICE)

				output = model(images)

				tdice = dice_loss(output, labels)

				test_running_loss += tdice.item()

		model_train_loss = train_running_loss / len(train_loader)
		model_test_loss  = test_running_loss  / len(test_loader)

		history['train_loss'].append(model_train_loss)
		history['test_loss'].append(model_test_loss)

		toc = time.time() - tic
		history['toc'].append(toc)
		print('Epoch {0} of {1}, Train Loss: {2:.4f}, Test Loss: {3:.4f}, Time: {4:.2f} sec'
			  .format(n+1, args.num_epoch, model_train_loss, model_test_loss, toc))

		if n % save_frequency == 0:
			save_model(args.model_file, model, optimizer, history, '{0:02d}'.format(n))

	save_model(args.model_file, model, optimizer, history)

	if args.log_file is not None:
		with open(args.log_file, 'w') as file:
			file.write('Epoch,Train Loss,Test Loss,Time\n')
			for n in range(args.num_epoch):
				file.write('{0},{1:.4f},{2:.4f},{3:.2f}\n'.format(
					n+1,
					history['train_loss'][n], history['test_loss'][n],
					history['toc'][n]
				))

	print(f'Training complete - model saved to {args.model_file}')

#----------------------------------------------------------
def test(args):
	test_data = Data(args.test_path)
	print('Test data:', len(test_data))

	num_batches = np.ceil(len(test_data) / num_batch).astype('int')

	model = BasicallyUnet(in_channels=1, base_channels=nfilter).to(DEVICE)
	checkpoint = torch.load(args.model_file, weights_only=True)
	model.load_state_dict(checkpoint['model'])
	del checkpoint

	test_loader = DataLoader(test_data, batch_size=num_batch, num_workers=num_workers)

	mse_fn = nn.MSELoss(reduction='mean')
	model.eval()

	test_running_loss = 0.0
	test_dice         = 0.0
	test_mse          = 0.0
	loss_array        = []
	dice_array        = []
	mse_array         = []

	with torch.no_grad():
		for i, (images, labels) in enumerate(test_loader):
			end = '\r' if i < len(test_loader) - 1 else '\n'
			print('Batch {0} of {1}...'.format(i + 1, num_batches), end=end, flush=True)

			images = images.to(DEVICE)
			labels = labels.to(DEVICE)

			output = model(images)

			tdice = dice_loss(output, labels)
			tmse  = mse_fn(output, labels)
			tloss = tdice

			test_running_loss += tloss.item()
			test_dice         += tdice.item()
			test_mse          += tmse.item()

			loss_array.append(tloss.item())
			dice_array.append(tdice.item())
			mse_array.append(tmse.item())

			# Save outputs
			for j in range(output.shape[0]):
				global_idx = i * num_batch + j

				pred_np  = output[j, 0].cpu().numpy()
				pred_np  = (pred_np * 255).astype(np.uint8)

				img_np   = images[j, 0].cpu().numpy()
				img_np   = (img_np * 255).astype(np.uint8)

				label_np = labels[j, 0].cpu().numpy()
				label_np = (label_np * 255).astype(np.uint8)

				cv2.imwrite(os.path.join(args.result_path, f'output_{global_idx:04d}_pred.png'),  pred_np)
				cv2.imwrite(os.path.join(args.result_path, f'output_{global_idx:04d}_image.png'), img_np)
				cv2.imwrite(os.path.join(args.result_path, f'output_{global_idx:04d}_label.png'), label_np)

	with open(os.path.join(args.result_path, 'loss.csv'), 'w') as file:
		file.write('Batch,Loss,Dice,MSE\n')
		for n in range(len(loss_array)):
			file.write('{0},{1:.4f},{2:.4f},{3:.4f}\n'.format(
				n + 1, loss_array[n], dice_array[n], mse_array[n]
			))

	print('Test Loss: {0:.4f}, Dice: {1:.4f}, MSE: {2:.5f}'.format(
		test_running_loss / len(test_loader),
		test_dice         / len(test_loader),
		test_mse          / len(test_loader)
	))
#----------------------------------------------------------

if __name__ == "__main__":

	print('\n Basically U-Neisnet - Training and Testing Script\n')

	parser = argparse.ArgumentParser(
		formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32),
		epilog='\nFor more information, please check README.md\n',
		exit_on_error=False
	)
	parser._optionals.title = 'command arguments'

	parser.add_argument('-job',        type=str, help='job name',                    required=True)
	parser.add_argument('-data_path',  type=str, help='image dataset directory',     metavar='')
	parser.add_argument('-test_path',  type=str, help='test dataset directory',      metavar='')
	parser.add_argument('-result_path',  type=str, help='results directory',      metavar='')
	parser.add_argument('-model_file', type=str, help='model weights file',          metavar='')
	parser.add_argument('-log_file',   type=str, help='log information filename',    metavar='')
	parser.add_argument('-num_epoch',  type=int, help='number of training epochs',   metavar='')
	parser.add_argument('-state_file', type=str, help='state file',                  metavar='')

	try:
		args = parser.parse_args()
	except SystemExit:
		raise ValueError('invalid parameters')

	if args.data_path  is not None: args.data_path  = os.path.abspath(args.data_path)
	if args.test_path  is not None: args.test_path  = os.path.abspath(args.test_path)
	if args.result_path  is not None: args.result_path  = os.path.abspath(args.result_path)
	if args.state_file is not None: args.state_file = os.path.abspath(args.state_file)

	#----------------------------------------------------------

	if args.job == 'TRAIN':
		if args.data_path  is None or not os.path.exists(args.data_path):  raise Exception('invalid data_path')
		if args.test_path  is None or not os.path.exists(args.test_path):  raise Exception('invalid test_path')
		if args.model_file is None: raise Exception('invalid model file')
		if args.log_file   is None: raise Exception('invalid log file')
		if args.num_epoch  is None: raise Exception('invalid epoch number')
		train(args)

	elif args.job == 'TEST':
		if args.test_path  is None or not os.path.exists(args.test_path):  raise Exception('invalid test_path')
		if args.model_file is None or not os.path.exists(args.model_file): raise Exception('invalid model file')
		test(args)

	else:
		print('Invalid job!')
		sys.exit(-1)

	print(args.job.capitalize() + ' job completed!')
