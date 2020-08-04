import os
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pylab as plt

def results_collect(init, scaled_init, bn, sp, act, depth):
	training_mean = 0
	training_std = 0
	res_mean = 0
	res_std = 0

	results_path = init + '_sp' + str(sp).replace('.', '_')
	if scaled_init:
		results_path += '_scaled'
	if bn:
		results_path += '_bn'

	results_path += '_' + act + '_' + str(depth)

	print('this is the path')
	print(results_path)
	print(os.getcwd())

	if not os.path.isdir(results_path):
		print(results_path, 'doesnt exist')
		return [0, 0, [0], [0], [0]]

	os.chdir(results_path)

	# if not os.path.isfile('best_acc.npy'):
	# 	print('dont have these results yet {}'.format(results_path))
	# 	return [0, 0, [0], [0], [0]]



	# res_vec = np.load('best_acc.npy')
	# training_acc = np.load('test_acc.npy')
	ratio_pruned = np.load('ratios_pruned.npy')

	# training_mean = training_acc.mean(0)
	# training_std = training_acc.std(0)


	#print(res_vec.shape, 'this is the shape of the res_vec (should be 3)')

	# res_mean = res_vec.mean()
	# res_std = res_vec.std()

	#print('We are using {}'.format(results_path))
	# print(res_mean)
	# print(res_std)

	os.chdir('../')
	return [res_mean, res_std, ratio_pruned, training_mean, training_std]


def plot_acc(tr_mean, tr_std):
	if _scaled_init:
		plt.plot(range(len(tr_mean)), tr_mean, label=init + '_scaled')
	else:
		plt.plot(range(len(tr_mean)), tr_mean, label=init)

	plt.fill_between(range(len(tr_mean)), tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)

	if init == 'EOC' and scaled_init == True:
		[res_mean, res_std, ratio, tr_mean, tr_std] = results_collect(init, False, bn, sp, act, depth)
		plt.plot(range(len(tr_mean)), tr_mean, label=init)
		plt.fill_between(range(len(tr_mean)), tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)



if __name__ == '__main__':
	# Move to the summary directory
	arch = 'vgg19'
	_plot_acc = False
	_plot_ratio = True
	pics_dir = '/data/ziz/ton/one_shot_pruning/GraSP/utils/{}_pics/'.format(arch)
	if not os.path.isdir(pics_dir): os.mkdir(pics_dir)
	if arch == 'resnet32':
		exp_name = 'cifar10_resnet32_SNIP'
		depth = 32
	elif arch == 'vgg19':
		#exp_name = 'cifar10_vgg19_SNIP_no_BN'
		exp_name = 'cifar10_vgg19_SNIP_circular'
		depth = 19

	dataset = 'cifar10'
	path_to_summary = '/data/ziz/ton/one_shot_pruning/GraSP/runs/pruning/{}/{}/{}/summary/'.format(dataset, arch,
	                                                                                               exp_name)
	os.chdir(path_to_summary)

	print(path_to_summary)

	# Collect the results
	# init = 'EOC'
	# scaled_init = False
	# bn = False
	# sp = 0.999
	#
	# res_mean, res_std = results_collect(init, scaled_init, bn, sp)

	init_vec = ['EOC', 'ordered', 'xavier'] #'kaiming',
	scaled_init = False
	bn = False
	#sp_vec = [0.0, 0.25, 0.50, 0.75, 0.98, 0.999]
	sp_vec = [0.95, 0.995]
	act_vec = ['tanh']


	results_matrix_mean = np.zeros([len(sp_vec), len(init_vec)])
	results_matrix_std = np.zeros([len(sp_vec), len(init_vec)])

	for act in act_vec:
		j = 0
		for sp in sp_vec:
			plt.figure()
			k = 0
			for init in init_vec:
				if init != 'EOC':
					_scaled_init = False
				else:
					_scaled_init = scaled_init

				[res_mean, res_std, ratio, tr_mean, tr_std] = results_collect(init, _scaled_init, bn, sp, act, depth)
				results_matrix_mean[j, k] = res_mean#.round(2)
				results_matrix_std[j, k] = res_std#.round(2)
				k += 1

				print('this is the ratio')
				print(ratio)
				if _plot_acc:
					if len(tr_mean) == 1:
						print('we continue')
						continue
					plot_acc(tr_mean, tr_std)
				elif _plot_ratio:
					if _scaled_init:
						plt.plot(ratio, label=init + '_scaled')
					else:
						plt.plot(ratio, label=init)

			if _plot_acc:
				#plt.ylim([25, 95])
				plt.legend()
				plt.title('ACC: We are using sp {} with act {}'.format(sp, act))
				if not os.path.isdir(pics_dir+'/{}'.format(act)): os.mkdir(pics_dir+'/{}'.format(act))
				plt.savefig('/data/ziz/ton/one_shot_pruning/GraSP/utils/{}_pics/{}/{}sp_act{}.png'.format(arch, act,sp, act))
			elif _plot_ratio:
				plt.legend()
				plt.title('Pruning Ratio: We are using sp {} with act {}'.format(sp, act))
				if not os.path.isdir(pics_dir+'/{}_ratio'.format(act)): os.mkdir(pics_dir+'/{}_ratio'.format(act))
				plt.savefig('/data/ziz/ton/one_shot_pruning/GraSP/utils/{}_pics/{}_ratio/{}sp_act{}.png'.format(arch, act,sp, act))
			j += 1

		#scaled_init = False
		print('\n')
		print('We are using act {}'.format(act))
		if scaled_init:
			print('We are using SCALED EOC')

		print('mean results {} x {}'.format(init_vec, sp_vec))
		print(results_matrix_mean)
		print('\n')
		print('std results {} x {}'.format(init_vec, sp_vec))
		print(results_matrix_std)
