from pyspark import SparkContext, SparkConf, RDD
import numpy as np
import math
from collections import defaultdict
import hashlib
import functools
import os
from arl.util.testing_support import *
from arl.imaging.facets import predict_facets
from arl.imaging.base import predict_skycomponent_visibility
from test import *
from Constants import *

os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
global bytebuffer
class Result():
	def __init__(self, data, hash):
		self.data = data
		self.hash = hash
# TODO 代码需要重新按照hash修改

def SDPPartitioner_pharp_alluxio(key):
	'''
		Partitioner_function
	'''
	return int(str(key).split(',')[2])

def SDPPartitioner(key):
	'''
		Partitioner_function
	'''
	return int(str(key).split(',')[4])

def MapPartitioner(partitions):
	def _inter(key):
		partition = partitions
		return partition[key]
	return _inter

def hash(st):
	temp = hashlib.md5()
	temp.update(st)
	return int(temp.hexdigest()[0:7], 16)

def extract_lsm_handle():
	partitions = defaultdict(int)
	partition = 0
	initset = []
	beam = 0
	major_loop = 0
	partitions[(beam, major_loop)] = partition
	partition += 1
	initset.append(((beam, major_loop), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(extract_lsm_kernel, True)

def local_sky_model_handle():
	partitions = defaultdict(int)
	partition = 0
	initset = []
	partitions[()] = partition
	partition += 1
	initset.append(((), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(local_sky_model_kernel, True)

def telescope_management_handle():
	partitions = defaultdict(int)
	partition = 0
	initset = []
	partitions[()] = partition
	partition += 1
	initset.append(((), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(telescope_management_kernel, True)

def visibility_buffer_handle(vis):
	initset = []
	beam = 0
	for frequency in range(0, 20):
		time = 0 
		baseline = 0
		polarisation = 0
		initset.append((beam, frequency, time, baseline, polarisation))
	return sc.parallelize(initset).map(lambda x: visibility_buffer_kernel(x, vis))

def telescope_data_handle(telescope_management):
	partitions = defaultdict(int)
	partition = 0
	dep_telescope_management = defaultdict(list)
	beam = 0
	frequency = 0
	time = 0
	baseline = 0
	partitions[(beam, frequency, time, baseline)] = partition
	partition += 1
	dep_telescope_management[()] = [(beam, frequency, time, baseline)]
	input_telescope_management = telescope_management.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]),dep_telescope_management[ix_data[0]]))
	partitioner = MapPartitioner(partitions)
	return input_telescope_management.partitionBy(len(partitions), partitioner).mapPartitions(telescope_data_kernel, True)

def reppre_ifft_handle(broadcast_lsm, image_graph):
	initset = []
	dep_extract_lsm = defaultdict(list)
	beam = 0
	major_loop = 0

	for frequency in range(0, 5):
		time = 0
		for facet in range(0, PIECE):
			for polarisation in range(0, 4):
				initset.append((beam, major_loop, frequency, time, facet, polarisation))

	return sc.parallelize(initset).map(lambda ix: reppre_ifft_kernel((ix, broadcast_lsm), image_graph))

def degrid_handle(reppre_ifft, broads_input_telescope_data):
	return reppre_ifft.flatMap(lambda ix: degrid_kernel((ix, broads_input_telescope_data)))

def pharotpre_dft_sumvis_handle(degrid, broadcast_lsm):
	return degrid.partitionBy(20, SDPPartitioner_pharp_alluxio).mapPartitions(lambda ix: pharotpre_dft_sumvis_kernel((ix, broadcast_lsm)))

def timeslots_handle(broads_input0, broads_input1):
	initset = []
	beam = 0
	for time in range(0, 120):
		frequency = 0
		baseline = 0
		polarisation = 0
		major_loop = 0
		initset.append((beam, major_loop, frequency, time, baseline, polarisation))

	return sc.parallelize(initset, 24).map(lambda ix: timeslots_kernel((ix, broads_input0, broads_input1)))

def solve_handle(timeslots):
	dep_timeslots = defaultdict(list)
	beam = 0
	major_loop = 0
	baseline = 0
	frequency = 0
	for time in range(0, 120):
		polarisation = 0
		dep_timeslots[(beam, major_loop, frequency, time, baseline, polarisation)] = (beam, major_loop, frequency, time, baseline, polarisation)
	return timeslots.map(solve_kernel)

def cor_subvis_flag_handle(broads_input0, broads_input1, broads_input2):
	initset = []
	beam = 0
	for frequency in range(0, 20):
		time = 0
		baseline = 0
		polarisation = 0
		major_loop = 0
		initset.append((beam, major_loop, frequency, time, baseline, polarisation))
	return sc.parallelize(initset, 20).map(lambda ix: cor_subvis_flag_kernel((ix, broads_input0, broads_input1, broads_input2)))

def grikerupd_pharot_grid_fft_rep_handle(broads_input_telescope_data, broads_input):
	initset = []
	beam = 0
	frequency = 0
	for facet in range(0, 81):
		for polarisation in range(0, 4):
			time = 0
			major_loop = 0
			initset.append((beam, major_loop, frequency, time, facet, polarisation))
	return sc.parallelize(initset).map(lambda ix: grikerupd_pharot_grid_fft_rep_kernel((ix, broads_input_telescope_data, broads_input)))

def sum_facets_handle(grikerupd_pharot_grid_fft_rep):
	initset = []
	beam = 0
	frequency = 0
	for facet in range(0, 81):
		for polarisation in range(0, 4):
			time = 0
			major_loop = 0
			initset.append((beam, major_loop, frequency, time, facet, polarisation))
	return grikerupd_pharot_grid_fft_rep.map(sum_facets_kernel)

def identify_component_handle(sum_facets):
	partitions = defaultdict(int)
	partition = 0
	dep_sum_facets = defaultdict(list)
	beam = 0
	major_loop = 0
	frequency = 0
	for facet in range(0, 81):
		partitions[(beam, major_loop, frequency, facet)] = partition
		partition += 1
		for i_polarisation in range(0, 4):
			dep_sum_facets[(beam, major_loop, frequency, 0, facet, i_polarisation)] = [(beam, major_loop, frequency, facet)]
	return sum_facets.partitionBy(81, SDPPartitioner).mapPartitions(identify_component_kernel_partitions)

def source_find_handle(identify_component):
   	partitions = defaultdict(int)
   	partition = 0
   	dep_identify_component = defaultdict(list)
   	beam = 0
   	major_loop = 0
   	partitions[(beam, major_loop)] = partition
   	partition += 1
   	for i_facet in range(0, 81):
   		dep_identify_component[(beam, major_loop, 0, i_facet)] = [(beam, major_loop)]
   	input_identify_component = identify_component.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_identify_component[ix_data[0]]))
   	partitioner = MapPartitioner(partitions)
   	return input_identify_component.partitionBy(len(partitions), partitioner).mapPartitions(source_find_kernel, True)

def subimacom_handle(sum_facets, identify_component):
	partitions = defaultdict(int)
	partition = 0
	dep_identify_component = defaultdict(list)
	dep_sum_facets = defaultdict(list)
	beam = 0
	major_loop = 0
	frequency = 0
	for facet in range(0, 81):
		partitions[(beam, major_loop, frequency, facet)] = partition
		partition += 1
		for polarisation in range(0, 4):
			dep_sum_facets[(beam, major_loop, frequency, 0, facet, polarisation)] = (beam, major_loop, frequency, facet)
		dep_identify_component[(beam, major_loop, frequency, facet)] = (beam, major_loop, frequency, facet)
	input_identify_component = identify_component.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_identify_component[ix_data[0]]))
	input_sum_facets = identify_component.flatMap(lambda ix_data: map(lambda  x: (x, ix_data[1]), dep_sum_facets[ix_data[0]]))
	partitioner = MapPartitioner(partitions)
	return input_identify_component.partitionBy(len(partitions), partitioner).cogroup(input_sum_facets.partitionBy(len(partitions), partitioner)).mapPartitions(subimacom_kernel)

def update_lsm_handle(local_sky_model, source_find):
	partitions = defaultdict(int)
	partition = 0
	dep_local_sky_model = defaultdict(list)
	dep_source_find = defaultdict(list)
	beam = 0
	major_loop = 0
	partitions[(beam, major_loop)] = partition
	partition += 1
	dep_local_sky_model[()] = [(beam, major_loop)]
	dep_source_find[(beam, major_loop)] = [(beam, major_loop)]
	input_local_sky_model = local_sky_model.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_local_sky_model[ix_data[0]]))
	input_source_find = source_find.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_source_find[ix_data[0]]))
	partitioner = MapPartitioner(partitions)
	# print 100*'-'
	# print input_source_find.cache().collect()
	# print input_local_sky_model.cache().collect()
	# print input_local_sky_model.partitionBy(len(partitions), partitioner).cogroup(input_source_find.partitionBy(len(partitions), partitioner)).collect()
	# print 100*'-'
	return input_local_sky_model.partitionBy(len(partitions), partitioner).cogroup(input_source_find.partitionBy(len(partitions), partitioner)).mapPartitions(update_lsm_kernel, True)

# kernel函数
def extract_lsm_kernel(ixs):
	'''
    	生成skycomponent(s)
    :param ixs: key
    :return: iter[(key, skycoponent)]
    '''
	Hash = 0
	input_size = 0

	# label = "Extract_LSM (0.0 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	# Hash ^= hash(label.encode("utf8"))
	# print(label + "(Hash " + hex(Hash) +  " from " + str((input_size / 1000000)) + " MB input")
	result = []
	for ix in ixs: #每次循环生成一个skycomponent
		f = numpy.array([100.0, 110.0, 120.0, 130.0])  # npol = 4
		flux = numpy.array([f, f, f, f, f])  # nchan,npol = 5, 4
		frequency = numpy.array([1e8, 1.1e8, 1.2e8, 1.3e8, 1.4e8])
		compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
		comp = create_skycomponent(flux=flux, frequency=frequency, direction=compabsdirection,
						   polarisation_frame=PolarisationFrame('stokesIQUV'))
		result.append((ix, comp))
	label = "Ectract_LSM (0.0M MB, 0.00 Tflop) " + str(ix)
	print(label + str(result))
	return iter(result)

def local_sky_model_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = next(ixs)[0]
	label = "Local Sky Model (0.0 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (Hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 0)), int)
	result[0] = Hash
	return iter([(ix, result)])

def telescope_management_kernel(ixs):
	'''
    	生成总的visibility类，留待telescope_data_kernel进一步划分
    :param ixs:
    :return: iter[(key, visibility)]
    '''
	# TODO 为何生成迭代器
	ix = next(ixs)[0]
	Hash = 0
	input_size = 0
	result = []
	lowcore = create_named_configuration('LOWBD2-CORE')
	times = numpy.linspace(-3, +3, 5) * (numpy.pi / 12.0)  # time = 13
	frequency = numpy.array([1e8, 1.1e8, 1.2e8, 1.3e8, 1.4e8])  # nchan = 5
	channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])  # nchan = 5
	phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
	vis = create_visibility(lowcore, times=times, frequency=frequency,
                            channel_bandwidth=channel_bandwidth,
                            phasecentre=phasecentre, weight=1,
                            polarisation_frame=PolarisationFrame('stokesIQUV'),
                            integration_time=1.0)  # vis [baselines, times, nchan, npol]

	label = "Telescope Management (0.0 MB, 0.00 Tflop) "
	result = (ix, vis)
	print(label + str(result))
	return iter([result])

def visibility_buffer_kernel(ixs, vis):
	'''
		按frequency生成list[visibility_buffer]
    :param ixs:
    :return:
    '''
	Hash = 0
	input_size = 0

	beam, frequency, time, baseline, polarisation = ixs
	actual_frequency = frequency // 4
	nvis = vis.data.shape[0] // 5
	viss = visibility_for_para(vis.data['vis'][actual_frequency::5], vis.data['uvw'][actual_frequency::5]
                            ,vis.data['time'][actual_frequency::5],vis.data['frequency'][actual_frequency::5],vis.data["channel_bandwidth"][actual_frequency::5]
                            ,vis.data['integration_time'][actual_frequency::5],vis.data['antenna1'][actual_frequency::5],vis.data['antenna2'][actual_frequency::5]
                            ,vis.data['weight'][actual_frequency::5], vis.data["imaging_weight"][actual_frequency::5], vis, "chan", nvis)
	result = (ixs, viss)
	label = "Visibility Buffer (546937.1 MB, 0.00 Tflop) "
	print(label + str(result))
	return result

def telescope_data_kernel(ixs):
	'''
		分割visibility类为visibility_para
    :param ixs:
    :return: iter[(key, visibility_para)]
    '''
	# TODO 为何只有(0,0,0,0)被取出 未按照原来的逻辑
	for data in ixs:
		ix, vis = data
		ufrequency = numpy.unique(vis.data['frequency'])
		m = {}
		for idx, chan in enumerate(ufrequency):
			m[chan] = idx
		utime = numpy.unique(vis.data['time'])
		t = {}
		for idx, time in enumerate(utime):
			t[time] = idx
		result = []
		for i in range(vis.nvis):
			v = visibility_for_para(vis.data['vis'][i], vis.data['uvw'][i]
                                      ,vis.data['time'][i],vis.data['frequency'][i],vis.data["channel_bandwidth"][i]
                               ,vis.data['integration_time'][i],vis.data['antenna1'][i],vis.data['antenna2'][i]
                               ,vis.data['weight'][i], vis.data["imaging_weight"][i], vis, "npol")
			result.append(((BEAM, m[v.data['frequency'][0]], t[v.data['time'][0]], v.data['antenna1'][0], v.data['antenna2'][0]), v))
			label = "Telescope Data (0.0 MB, 0.00 Tflop) "
	print(label, str(result))
	return iter(result)

def reppre_ifft_kernel(ixs, image_graph):
	'''

	:param ixs: (reppre(key), skycomponent(value)
	:return: (key, image_for_para)
	'''
	reppre, data_extract_lsm = ixs
	Hash = 0
	input_size = 0
	ix = reppre
	###生成测试数据============###
	image = copy.deepcopy(image_graph)
	ims, img_share = image_to_image_para(image, FACETS)
	beam, major_loop, frequency, time, facet, polarisation = ix
	im = ims[frequency * (NPOL * PIECE) + polarisation * (PIECE) + facet]
	# new wcs
	wcs = im.wcs.deepcopy()
	wcs.wcs.cdelt[0] = -0.001 * 180 / np.pi
	wcs.wcs.cdelt[1] = 0.001 * 180 / np.pi
	newshape2 = np.array(im.data.shape)
	newshape2[0] /= 2
	newshape2[1] /= 2
	###==============###
	for dix, comp in data_extract_lsm.value:
		insert_skycomponent_para(im, comp)
		# im = reproject_image_para(im, wcs, newshape2)[0]
		# im.data = fft(im.data)
	result = (ix, im)
	label = "Reprojection Predict + IFFT (14645.6 MB, 2.56 Tflop) "
	print(label + str(result))
	return result

def degrid_kernel(ixs):
	data_reppre_ifft, data_telescope_data = ixs
	Hash = 0
	input_size = 0
	iix, image = data_reppre_ifft
	ix = iix
	chan = ix[2]
	result = []
	for tix, conf in data_telescope_data.value:
		if chan == tix[1]:
			temp = predict_facets_para(conf, image)
			# (beam, major_loop, chan, time, facet, polarisation, a1, a2)
			result.append(((ix[0], ix[1], ix[2], tix[2], ix[4], ix[5], tix[3], tix[4]),temp))

	label = "Degridding Kernel Update + Degrid (674.8 MB, 0.59 Tflop) "
	# 复制四份
	mylist = np.empty(4, list)
	temp1 = ix[2] * 4
	mylist[0] = ((ix[0], ix[1], temp1, ix[3], ix[4], ix[5]), iter(copy.deepcopy(result)))
	temp2 = ix[2] * 4 + 1
	mylist[1] = ((ix[0], ix[1], temp2, ix[3], ix[4], ix[5]), iter(copy.deepcopy(result)))
	temp3 = ix[2] * 4 + 2
	mylist[2] = ((ix[0], ix[1], temp3, ix[3], ix[4], ix[5]), iter(copy.deepcopy(result)))
	temp4 = ix[2] * 4 + 3
	mylist[3] = ((ix[0], ix[1], temp4, ix[3], ix[4], ix[5]), iter(copy.deepcopy(result)))
	print(label + str(result))
	return (mylist)

def pharotpre_dft_sumvis_kernel(ixs):
	data_degkerupd_deg, data_extract_lsm = ixs
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
    # TODO newphasecentre从哪儿来？
	newphasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    # TODO frequency相同的visibility怎样合并？先合并在运算还是先运算再合并
	viss = []
	for dix, datas in data_degkerupd_deg:
		for data in datas:
            # (beam, major_loop, chan, time, facet, polarisation, a1, a2)
			viss.append(((dix[0], dix[1], dix[2], data[0][3], dix[4], dix[5], data[0][6], data[0][7]), phaserotate_visibility_para(data[1], newphasecentre, tangent=False)))
			ix = dix

	for dix, data in data_extract_lsm.value:
		predict_skycomponent_visibility_para(viss, data)

	label = "Phase Rotation Predict + DFT + Sum visibilities (546937.1 MB, 512.53 Tflop) "
	print(label + str(viss))
	# TODO 最后return的是visibility迭代器还是已经合并后的visibility
	return iter(viss)

def timeslots_kernel(ixs):
	idx, data_pharotpre_dft_sumvis, data_visibility_buffer = ixs
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	ix = idx
	label = "Timeslots (1518.3 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 5060952)), int)
	result[0] = Hash

	return (ix, result)

def solve_kernel(ixs):
	dix, data = ixs
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	Hash ^= data[0]
	input_size += data.shape[0]
	ix = dix
	label = "Solve (8262.8 MB, 16.63 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 27542596)), int)
	result[0] = Hash
	return (ix, result)

def cor_subvis_flag_kernel(ixs):
	ix, data_pharotpre_dft_sumvis, data_visibility_buffer, data_solve = ixs
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	label = "Correct + Subtract Visibility + Flag (153534.1 MB, 4.08 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 511780275)), int)
	result[0] = Hash
	return (ix, result)

def grikerupd_pharot_grid_fft_rep_kernel(ixs):
	idx, data_telescope_data, data_cor_subvis_flag = ixs
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	ix = idx
	label = "Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection (14644.9 MB, 20.06 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 48816273)), int)
	result[0] = Hash
	return (ix, result)

def sum_facets_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	dix, data = ixs
	Hash ^= data[0]
	input_size += data.shape[0]
	ix = dix

	label = "Sum Facets (14644.9 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 48816273)), int)
	result[0] = Hash
	return (ix, result)

def identify_component_kernel_partitions(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	for dix, data in ixs:
		Hash ^= data[0]
		input_size += data.shape[0]
		ix = dix
	label = "Identify Component (0.2 MB, 1830.61 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 533)), int)
	result[0] = Hash
	return iter([((ix[0], ix[1], ix[2], ix[4]), result)])

def source_find_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0)
	for dix, data in ixs:
		Hash ^= data[0]
		input_size += data.shape[0]
		ix = dix
	label = "Source Find (5.8 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 19200)), int)
	result[0] = Hash
	return iter([(ix, result)])

def subimacom_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0)
	# for temp in ixs:
	# 	ix, (data_identify_component, data_sum_facets) = temp
	# 	for data in data_identify_component:
	# 		Hash ^= data[0]
	# 		input_size += 1
	# 	for data in data_sum_facets:
	# 		Hash ^= data[0]
	# 		input_size += 1

	label = "Subtract Image Component (73224.4 MB, 67.14 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 244081369)), int)
	result[0] = Hash
	return iter([(ix, result)])

def update_lsm_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0)
	for temp in ixs:

		ix, (data_local_sky_mode, data_source_find) = temp
		for data in data_local_sky_mode:
			Hash ^= data[0]
			input_size += 1

		for data in data_source_find:
			Hash ^= data[0]
			input_size += 1

	label = "Update LSM (0.0 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 0)), int)
	result[0] = Hash
	return iter([(ix, result)])

scale_data = 0
scale_compute = 0
def make_Image(ims, image_graph):
	'''
		还原图片
	:param ims:
	:param image_graph:
	:return:
	'''
	im = []
	newshape2 = np.array(ims[0, 0, 0].data.shape)
	_, img_share = image_to_image_para(image_graph, FACETS)
	for i in range(NCHAN):
		for j in range(NPOL):
			temp = ims[i, :, j]
			im.append(image_gather(temp, FACETS, ims[i, 0, j].wcs,
								   np.zeros(newshape2[0:2] * FACETS, dtype=ims[i, 0, j].data.dtype)))
	return image_para_to_image(im, img_share)  # 还原为原Image类，查看是否相等

if __name__ == '__main__':
	conf = SparkConf().setAppName("SDP Pipeline").setMaster("local[1]")
	sc = SparkContext(conf=conf)
	lowcore = create_named_configuration('LOWBD2-CORE')
	times = numpy.linspace(-3, +3, 5) * (numpy.pi / 12.0)  # time = 13
	frequency = numpy.array([1e8, 1.1e8, 1.2e8, 1.3e8, 1.4e8])  # nchan = 5
	channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])  # nchan = 5
	phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
	f = numpy.array([100.0, 110.0, 120.0, 130.0])  # npol = 4
	flux = numpy.array([f, f, f, f, f])  # nchan,npol = 5, 4
	compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
	comp = create_skycomponent(flux=flux, frequency=frequency, direction=compabsdirection,
							   polarisation_frame=PolarisationFrame('stokesIQUV'))  # flux [2,4]
	image_graph = create_test_image(frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
									polarisation_frame=PolarisationFrame('stokesIQUV'))  # data [5,4,256,256]
	vis = create_visibility(lowcore, times=times, frequency=frequency,
							channel_bandwidth=channel_bandwidth,
							phasecentre=phasecentre, weight=1,
							polarisation_frame=PolarisationFrame('stokesIQUV'),
							integration_time=1.0)  # vis [baselines, times, nchan, npol]
	vis_share = visibility_share(vis)

	# 串行程序综合测试
	newwcs = image_graph.wcs.deepcopy()
	newwcs.wcs.cdelt[0] = -0.001 * 180 / np.pi
	newwcs.wcs.cdelt[1] = 0.001 * 180 / np.pi
	newshape = np.array(image_graph.data.shape)
	newshape[2] /= 2
	newshape[3] /= 2
	newphasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-10.0 * u.deg, frame='icrs', equinox='J2000')
	image = copy.deepcopy(image_graph)
	new_vis = copy.deepcopy(vis)
	insert_skycomponent(image, comp)
	# TODO FACET为1时，并行和串行的结果相同，但FACET不为1时结果不同，为了便于后面的验证，因此不再进行这两步操作
	# image = reproject_image(image, newwcs, newshape)[0]
	# image.data = fft(image.data)
	predict_facets(new_vis, image)
	new_vis = phaserotate_visibility(new_vis, newphasecentre, tangent=False)
	predict_skycomponent_visibility(new_vis, comp)




	# === Extract Lsm ===
	extract_lsm = extract_lsm_handle()
	broadcast_lsm = sc.broadcast(extract_lsm.collect())
	# === Local Sky Model ===
	local_sky_model = local_sky_model_handle()
	# === Telescope Management ===
	telescope_management = telescope_management_handle()
	# # === Visibility Buffer ===
	visibility_buffer = visibility_buffer_handle(vis)
	visibility_buffer.cache()
	broads_input1 = sc.broadcast(visibility_buffer.collect())
	# === reppre_ifft ===
	reppre_ifft = reppre_ifft_handle(broadcast_lsm, image_graph)
	reppre_ifft.cache()

	# TODO 验证reppre_ifft的正确性，通过
	# im = make_Image(np.array(reppre_ifft.collect())[:, 1].reshape([NCHAN, PIECE, NPOL]), image_graph)
	# image_right(im, image)

	# === Telescope Data ===
	telescope_data = telescope_data_handle(telescope_management)
	broads_input_telescope_data = sc.broadcast(telescope_data.collect())
	# # === degrid ===
	degrid = degrid_handle(reppre_ifft, broads_input_telescope_data)
	degrid.cache()

	# TODO 验证degrid的正确性，通过
	# viss = degrid.collect()
	# map = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
	# visibility = copy.deepcopy(vis)
	# for v in viss:
	# 	if(v[0][2] % 4 != 0):
	# 		continue
	# 	for i in v[1]:
	# 		beam, major_loop, chan, time, facet, pol, an1, an2 = i[0]
	# 		nvis = time * NCHAN * NBASE + map[(an1, an2)] * NCHAN + chan
	# 		visibility.data['vis'][nvis] += i[1].data['vis'][0]
    #
    #
	# visibility_right(visibility, new_vis)

	# === pharotpre_dft_sumvis ===
	pharotpre_dft_sumvis = pharotpre_dft_sumvis_handle(degrid, broadcast_lsm)
	pharotpre_dft_sumvis.cache()
	# TODO 验证pharotpre_dft_sumvis的正确性, 待完成

	broads_input0 = sc.broadcast(pharotpre_dft_sumvis.collect())
	# # === Timeslots ===
	# timeslots = timeslots_handle(broads_input0, broads_input1)
	# timeslots.cache()
	# # === solve ===
	# solve = solve_handle(timeslots)
	# solve.cache()
	# broads_input2 = sc.broadcast(solve.collect())
	# # === correct + Subtract Visibility + Flag ===
	# cor_subvis_flag = cor_subvis_flag_handle(broads_input0, broads_input1, broads_input2)
	# cor_subvis_flag.cache()
	# broads_input = sc.broadcast(cor_subvis_flag.collect())
	# # === Gridding Kernel Update + Phase Rotation + Grid + FFT + Rreprojection ===
	# grikerupd_pharot_grid_fft_rep = grikerupd_pharot_grid_fft_rep_handle(broads_input_telescope_data, broads_input)
	# grikerupd_pharot_grid_fft_rep.cache()
	# # ===Sum Facets ===
	# sum_facets = sum_facets_handle(grikerupd_pharot_grid_fft_rep)
	# sum_facets.cache()
	# # === Identify Component ===
	# identify_component = identify_component_handle(sum_facets)
	# # === Source Find ===
	# source_find = source_find_handle(identify_component)
	# # === Substract Image Component ===
	# subimacom = subimacom_handle(sum_facets, identify_component)
	# # === Update LSM ===
	# update_lsm = update_lsm_handle(local_sky_model, source_find)
	# # # === Terminate ===
	# print("Finishing...")
	# #print("Subtract Image Component: %d" % subimacom.count())
	# print("Update LSM: %d" % update_lsm.count())



