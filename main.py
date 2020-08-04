import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.base.init_utils import weights_init_EOC, weights_init_kaiming_xavier, weights_init_xavier, \
	weights_init_kaiming_relu, weights_init_kaiming_tanh, weights_init_ord
from models.model_base import ModelBase

from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_ratio', type=float)
	parser.add_argument('--iteration', type=int)
	parser.add_argument('--reset_to', type=int)

	parser.add_argument('--network', type=str, default='vgg')
	parser.add_argument('--dataset', type=str, default='cifar10')
	parser.add_argument('--depth', type=int, default=19)

	parser.add_argument('--pretrain_model', type=str)
	parser.add_argument('--log_dir', type=str, default='runs/')


def init_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--run', type=str, default='')
	parser.add_argument('--init', type=str, default='kaiming_xavier')
	parser.add_argument('--target_ratio', type=float, default=0.)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--scaled', action='store_true')
	parser.add_argument('--bn', action='store_true')
	parser.add_argument('--act', type=str, default='relu')
	args = parser.parse_args()
	runs = None
	if len(args.run) > 0:
		runs = args.run
	config = process_config(args.config, runs)

	os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(args.gpu)

	return config, args


def init_logger(config, args):
	makedirs(config.summary_dir)
	makedirs(config.checkpoint_dir)

	# set logger
	path = os.path.dirname(os.path.abspath(__file__))
	path_model = os.path.join(path, 'models/base/%s.py' % config.network.lower())
	path_main = os.path.join(path, 'main.py')
	#path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
	logger = get_logger('log', logpath=config.summary_dir + '/',
	                    filepath=path_model, package_files=[path_main])
	logger.info(dict(config))
	summary_writer_path = config.summary_dir + '/' + args.init 
	if args.scaled:
		summary_writer_path += '_scaled'
	if args.bn:
		summary_writer_path += '_bn'

	summary_writer_path += '_'+ args.act + '_' + str(config.depth)

	writer = SummaryWriter(summary_writer_path)
	# sys.stdout = open(os.path.join(config.summary_dir, 'stdout.txt'), 'w+')
	# sys.stderr = open(os.path.join(config.summary_dir, 'stderr.txt'), 'w+')
	return logger, writer


def print_mask_information(mb, logger):
	ratios = mb.get_ratio_at_each_layer()
	logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
	count = 0
	ratio_vec = []
	for k, v in ratios.items():
		if k == 'ratio':
			continue
		logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
		count += 1
		ratio_vec.append(v)
		print(count)
	return ratio_vec



def save_state(net, acc, epoch, loss, config, ckpt_path, is_best=False):
	print('Saving..')
	state = {
		'net': net,
		'acc': acc,
		'epoch': epoch,
		'loss': loss,
		'args': config
	}
#	if not is_best:
		#torch.save(state, '%s/pruned_%s_%s%s_%d.t7' % (ckpt_path,
		                                           #    config.dataset,
		                                            #   config.network,
		                                             #  config.depth,
		                                              # epoch))
#	else:
		#torch.save(state, '%s/finetuned_%s_%s%s_best.t7' % (ckpt_path,
		                                               #     config.dataset,
		                                                #    config.network,
		                                                 #   config.depth))


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0

	lr_scheduler(optimizer, epoch)
	desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
	        (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))

	writer.add_scalar('iter_%d/train/lr' % iteration, lr_scheduler.get_lr(optimizer), epoch)

	prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
	for batch_idx, (inputs, targets) in prog_bar:
		inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		# import pdb; pdb.set_trace()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
		        (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		prog_bar.set_description(desc, refresh=True)

	writer.add_scalar('iter_%d/train/loss' % iteration, train_loss / (batch_idx + 1), epoch)
	writer.add_scalar('iter_%d/train/acc' % iteration, 100. * correct / total, epoch)


def test(net, loader, criterion, epoch, writer, iteration):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
	        % (test_loss / (0 + 1), 0, correct, total))

	prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
	with torch.no_grad():
		for batch_idx, (inputs, targets) in prog_bar:
			inputs, targets = inputs.cuda(), targets.cuda()
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
			        % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
			prog_bar.set_description(desc, refresh=True)

	# Save checkpoint.
	acc = 100. * correct / total

	writer.add_scalar('iter_%d/test/loss' % iteration, test_loss / (batch_idx + 1), epoch)
	writer.add_scalar('iter_%d/test/acc' % iteration, 100. * correct / total, epoch)
	return acc


def train_once(mb, net, trainloader, testloader, writer, config, ckpt_path, learning_rate, weight_decay, num_epochs,
               iteration, logger, args):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
	lr_schedule = {0: learning_rate,
	               int(num_epochs * 0.25): learning_rate * 0.1,
	               int(num_epochs * 0.5): learning_rate * 0.01,
                       int(num_epochs * 0.75):learning_rate * 0.001}
	lr_scheduler = PresetLRScheduler(lr_schedule)
	best_acc = 0
	best_epoch = 0

	if args.target_ratio == None:
		target_ratio = config.target_ratio
	else:
		target_ratio = args.target_ratio

	test_acc_vec = []
	for epoch in range(num_epochs):
		train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, iteration=iteration)
		test_acc = test(net, testloader, criterion, epoch, writer, iteration)
		test_acc_vec.append(test_acc)
		if test_acc > best_acc:
			print('Saving..')
			state = {
				'net': net,
				'acc': test_acc,
				'epoch': epoch,
				'args': config,
			}
			path = os.path.join(ckpt_path, 'finetune_%s_%s%s_it%d_best.pth.tar' % (config.dataset,
			                                                                           config.network,
			                                                                           config.depth,
			                                                                           iteration))
			#torch.save(state, path)
			best_acc = test_acc
			best_epoch = epoch
	logger.info('Iteration [%d], best acc: %.4f, epoch: %d' %
	            (iteration, best_acc, best_epoch))

	return best_acc, test_acc_vec


def get_exception_layers(net, exception):
	exc = []
	idx = 0
	for m in net.modules():
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			if idx in exception:
				exc.append(m)
			idx += 1
	return tuple(exc)


def main(config, args):
	# init logger
	classes = {
		'cifar10': 10,
		'cifar100': 100,
		'mnist': 10,
		'tiny_imagenet': 200
	}
	logger, writer = init_logger(config, args)
	best_acc_vec = []
	test_acc_vec_vec = []

	for _ in range(1):
		# build model
		model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', args.bn),
		                    scaled=args.scaled, act=args.act)
		#mask = None
		mb = ModelBase(config.network, config.depth, config.dataset, model)
		mb.cuda()
		#if mask is not None:
		#	mb.register_mask(mask)
		#	ratio_vec_ = print_mask_information(mb, logger)

		# preprocessing
		# ====================================== get dataloader ======================================
		trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4)
		# ====================================== fetch configs ======================================
		ckpt_path = config.checkpoint_dir
		num_iterations = config.iterations
		if args.target_ratio == None:
			target_ratio = config.target_ratio
		else:
			target_ratio = args.target_ratio

		normalize = config.normalize
		# ====================================== fetch exception ======================================
		exception = get_exception_layers(mb.model, str_to_list(config.exception, ',', int))
		logger.info('Exception: ')

		for idx, m in enumerate(exception):
			logger.info('  (%d) %s' % (idx, m))

		# ====================================== fetch training schemes ======================================
		ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
		learning_rates = str_to_list(config.learning_rate, ',', float)
		weight_decays = str_to_list(config.weight_decay, ',', float)
		training_epochs = str_to_list(config.epoch, ',', int)
		logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
		            (normalize, num_iterations, target_ratio, ratio))
		logger.info('Basic Settings: ')
		for idx in range(len(learning_rates)):
			logger.info('  %d: LR: %.5f, WD: %.5f, Epochs: %d' % (idx,
			                                                      learning_rates[idx],
			                                                      weight_decays[idx],
			                                                      training_epochs[idx]))

		iteration = 0
		best_acc, test_acc_vec = train_once(mb=mb,
			                                net=mb.model,
			                                trainloader=trainloader,
			                                testloader=testloader,
			                                writer=writer,
			                                config=config,
			                                ckpt_path=ckpt_path,
			                                learning_rate=learning_rates[iteration],
			                                weight_decay=weight_decays[iteration],
			                                num_epochs=training_epochs[iteration],
			                                iteration=iteration,
			                                logger=logger,
			                                args=args)

		best_acc_vec.append(best_acc)
		test_acc_vec_vec.append(test_acc_vec)

		results_path = config.summary_dir + '/' + args.init
		if args.scaled:
			results_path += '_scaled'
		if args.bn:
			results_path += '_bn'

		results_path += '_' + args.act + '_' + str(config.depth)

		np.save(results_path + '/best_acc', np.array(best_acc_vec))
		np.save(results_path + '/test_acc', np.array(test_acc_vec_vec))



if __name__ == '__main__':
	torch.manual_seed(12)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	config, args = init_config()
	main(config, args)
