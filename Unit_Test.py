from test import *
from arl.imaging.base import predict_skycomponent_visibility
from arl.visibility.coalesce import decoalesce_visibility, coalesce_visibility
from arl.imaging.facets import predict_facets
from arl.calibration.solvers import solve_gaintable
import unittest
from collections import defaultdict
import numpy as np

lowcore = create_named_configuration('LOWBD2-CORE')
times = numpy.linspace(-3, +3, 5) * (numpy.pi / 12.0)  # time = 5
frequency = numpy.array([1e8, 1.1e8, 1.2e8, 1.3e8, 1.4e8])  # nchan = 5
channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])  # nchan = 5

# Define the component and give it some polarisation and spectral behaviour
f = numpy.array([100.0, 110.0, 120.0, 130.0])  # npol = 4
flux = numpy.array([f, f, f, f, f])  # nchan,npol = 2, 4

phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')

comp = create_skycomponent(flux=flux, frequency=frequency, direction=compabsdirection,
                           polarisation_frame=PolarisationFrame('linear'))  # flux [2,4]

image_graph = create_test_image(frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                                polarisation_frame=PolarisationFrame('linear'))  # data [2,4,256,256]
blockvis = create_blockvisibility(lowcore, times=times, frequency=frequency,
                                                 channel_bandwidth=channel_bandwidth,
                                                 phasecentre=phasecentre, weight=1,
                                                 polarisation_frame=PolarisationFrame('linear'),
                                                 integration_time=1.0)
# 创建用blockvis的vis
vis = coalesce_visibility(blockvis)
# vis = create_visibility(lowcore, times=times, frequency=frequency,
#                         channel_bandwidth=channel_bandwidth,
#                         phasecentre=phasecentre, weight=1,
#                         polarisation_frame=PolarisationFrame('stokesIQUV'),
#                         integration_time=1.0)  # vis [baselines, times, nchan, npol]



class TestImageIterators(unittest.TestCase):
    #===predict_module===
    # def test_visibility_class(self):
    #     for mode in ['pol', 'npol', 'chan']:
    #         viss, vis_share = visibility_to_visibility_para(vis, mode)
    #         new_vis = visibility_para_to_visibility(viss, mode, vis_share)
    #         visibility_right(vis, new_vis)
    #         print("%s visibility test passed" % mode)
    #
    # def test_image_class(self):
    #     ims, image_share = image_to_image_para(image_graph, FACETS)
    #     new_im = image_para_facet_to_image(ims, FACETS, image_share)
    #     image_right(image_graph, new_im)
    #     print("image test passed")
    #
    #
    # def test_insert_skycomponent(self):
    #     '''
    #         test insert_skycomponent
    #         已通过 2017，10，26
    #     :return:
    #     '''
    #     for method in ["Sinc", "Lanczos", "PSWF", "normal"]:
    #         ims, image_share = image_to_image_para(image_graph, FACETS)
    #         for i in ims:
    #             insert_skycomponent_para(i, comp, insert_method=method)
    #         new_im = image_para_facet_to_image(ims, FACETS, image_share)  # 还原为原Image类，查看是否相等
    #         insert_skycomponent(image_graph, comp, insert_method=method)
    #         image_right(image_graph, new_im)
    #         print("%s method test passed" % method)
    #
    #
    # def test_Reproject_image(self):
    #     '''
    #         test reproject_image
    #         已通过 2017, 11, 5
    #     :return:
    #     '''
    #     newwcs, newshape = create_new_wcs_new_shape(image_graph.wcs, image_graph.data.shape)
    #     ims, image_share = image_to_image_para(image_graph, FACETS)
    #
    #     ims2 = []
    #     for i in ims:
    #         wcs,newshape2 = create_new_wcs_new_shape(i.wcs, i.data.shape)
    #         ims2.append(reproject_image_para(i, wcs, newshape2)[0])
    #     new_im = image_para_facet_to_image(ims2, FACETS, image_share)
    #
        # im = reproject_image(image_graph, newwcs, newshape)[0]
    #     image_right(im, new_im)
    #
    # def test_predict_facets(self):
    #     '''
    #         test predict_facets
    #         已通过 2017, 11, 1
    #     :return:
    #     '''
    #
    #     viss, visibility_share = visibility_to_visibility_para(vis, 'npol')
    #     ims, image_share = image_to_image_para(image_graph, FACETS)
    #     viss2 = []
    #     for v in viss:
    #         for im in ims:
    #             if v[0][0] == im.frequency:
    #                 viss2.append(((v[0][0], v[0][1], v[0][2], v[0][3], im.facet, im.polarisation), predict_facets_para(v[1], im)))
    #     map = {(0,1): 0, (0,2): 1, (1,2):2}
    #
    #     new_vis = copy.deepcopy(vis)
    #     for i, v in enumerate(viss2):
    #         print(v[0],v[1].data['vis'])
    #         id = v[0][0] + v[0][1] * NBASE * NCHAN + map[(v[0][2], v[0][3])] * NCHAN
    #         new_vis.data['vis'][id] += v[1].data['vis'][0]
    #
    #     visibility = copy.deepcopy(vis)
    #     predict_facets(visibility, image_graph)
    #     print(visibility.data['vis'])
    #     visibility_right(visibility, new_vis)
    #
    # def test_phaserotate_visibility(self):
    #     '''
    #         test phaserotate_visibility
    #         已通过 2017 10 30
    #     :return:
    #     '''
    #     viss, visibility_share = visibility_to_visibility_para(vis, 'npol')
    #     newphasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-10.0 * u.deg, frame='icrs', equinox='J2000')
    #     viss2 = []
    #     for v in viss:
    #         viss2.append((v[0], phaserotate_visibility_para(v[1], newphasecentre, tangent=False)))
    #
    #     new_vis = visibility_para_to_visibility(viss2, 'npol', visibility_share)
    #
    #     visibility = copy.deepcopy(vis)
    #     visibility = phaserotate_visibility(visibility, newphasecentre, tangent=False)
    #     visibility_right(visibility, new_vis)
    #
    # def test_predict_skycomponent_visibility(self):
    #     '''
    #         test predict_skycomponent_visibility
    #         已通过 2017 10 30
    #     :return:
    #     '''
    #     viss, visibility_share = visibility_to_visibility_para(vis, 'pol')
    #     predict_skycomponent_visibility_para(viss, comp, mode="test")
    #     new_vis = visibility_para_to_visibility(viss, 'pol', visibility_share)
    #     visibility = copy.deepcopy(vis)
    #     predict_skycomponent_visibility(visibility, comp)
    #     visibility_right(visibility, new_vis)

    # def test_predict_module(self):
    #     '''
    #         将predict模块的方法串联起来进行测试
    #         fft和reproject对切片后的数据结果不一致
    #         predict_skycomponent_visibility也不一致，但是是因为切片被多次加总导致，该不一致可以很好的改正，使用predict_skycomponent_visibility_modified函数修正此问题
    #
    #     :return:
    #     '''
    #     newphasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-10.0 * u.deg, frame='icrs', equinox='J2000')
    #
    #     # ===串行===
    #     image = copy.deepcopy(image_graph)
    #     insert_skycomponent(image, comp, insert_method="Sinc")
    #     wcs, newshape = create_new_wcs_new_shape(image.wcs, image.data.shape)
    #     # image.data = fft(image.data)
    #     # image = reproject_image(image, wcs, newshape)[0]
    #     visibility = copy.deepcopy(vis)
    #     predict_facets(visibility, image)
    #     visibility = phaserotate_visibility(visibility, newphasecentre, tangent=False)
    #     predict_skycomponent_visibility(visibility, comp)
    #
    #     # ===并行===
    #     viss, visibility_share = visibility_to_visibility_para(vis, 'npol')
    #     ims, image_share = image_to_image_para(image_graph, FACETS)
    #     ims2 = []
    #     for i in ims:
    #         insert_skycomponent_para(i, comp, insert_method='Sinc')
    #         wcs, newshape2 = create_new_wcs_new_shape(i.wcs, i.data.shape)
    #         # i.data = fft(i.data)
    #         # ims2.append(reproject_image_para(i, wcs, newshape2)[0])
    #         ims2.append(i)
    #     # new_im = image_para_facet_to_image(ims2, FACETS, image_share)
    #
    #     viss2 = []
    #     for v in viss:
    #         for im in ims2:
    #             if v[0][0] == im.frequency:
    #                 temp = predict_facets_para(v[1], im)
    #                 viss2.append(((v[0][0], v[0][1], v[0][2], v[0][3], im.facet, im.polarisation), phaserotate_visibility_para(temp, newphasecentre, tangent=False)))
    #     map = {(0,1): 0, (0,2): 1, (1,2):2}
    #     # predict_skycomponent_visibility_para(viss2, comp, mode='test2')
    #     predict_skycoponent_visibility_para_modified(viss2, comp, mode='test2')
    #
    #     new_vis = copy.deepcopy(vis)
    #     for i, v in enumerate(viss2):
    #         id = v[0][0] + v[0][1] * NBASE * NCHAN + map[(v[0][2], v[0][3])] * NCHAN
    #         new_vis.data['vis'][id] += v[1].data['vis'][0]
    #         new_vis.data['uvw'][id] = v[1].data['uvw'][0]
    #
    #
    #     # ===比较===
    #     visibility_right(visibility, new_vis)

    #===calibration_module===
    def test_solve_gaintable(self):
        '''
            测试solve_gaintable
        :return:
        '''
        #===串行===
        newphasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-10.0 * u.deg, frame='icrs', equinox='J2000')
        image = copy.deepcopy(image_graph)
        insert_skycomponent(image, comp, insert_method="Sinc")

        visibility = copy.deepcopy(vis)
        predict_facets(visibility, image)

        #人工创建一个非空的modelvis
        model_vis = decoalesce_visibility(copy.deepcopy(visibility))
        visibility = phaserotate_visibility(visibility, newphasecentre, tangent=False)
        predict_skycomponent_visibility(visibility, comp)
        block_vis = decoalesce_visibility(visibility)

        gt = solve_gaintable(block_vis, model_vis)

        #===并行===
        viss, visibility_share = visibility_to_visibility_para(vis, 'npol')
        ims, image_share = image_to_image_para(image_graph, FACETS)
        ims2 = []
        for i in ims:
            insert_skycomponent_para(i, comp, insert_method='Sinc')
            ims2.append(i)

        model_vis1 = []
        viss2 = []
        for v in viss:
            for im in ims2:
                if v[0][0] == im.frequency:
                    temp = predict_facets_para(v[1], im)
                    # (chan, time, ant1,  ant2)
                    model_vis1.append(((v[0][0], v[0][1], v[0][2], v[0][3]), temp))
                    viss2.append(((v[0][0], v[0][1], v[0][2], v[0][3], im.facet, im.polarisation), phaserotate_visibility_para(temp, newphasecentre, tangent=False)))
        predict_skycoponent_visibility_para_modified(viss2, comp, mode='test2')
        # 将visibility的facet和polarisation合并起来
        viss3 = defaultdict(list)
        model_vis2 = defaultdict(list)
        for v in range(0,len(viss2),4 * 4):
            temp = copy.deepcopy(viss2[v][1])
            temp2 = copy.deepcopy(model_vis1[v][1])
            for id in range(v+1, v+16):
                temp.data['vis'] += viss2[id][1].data['vis']
                temp2.data['vis'] += model_vis1[id][1].data['vis']
            viss3[viss2[v][0][0:2]].append(((viss2[v][0][2:4]), temp))
            model_vis2[model_vis1[v][0][0:2]].append(((model_vis1[v][0][2:]), temp2))



        # TODO 将并行程序从此开始填入
        gts = []
        for key in viss3:
            xs = []
            xwts = []
            for v, mv in zip(viss3[key], model_vis2[key]):
                x, xwt = solve_gaintable_para(v[1], mv[1])
                xs.append(((0,0,0,0,key[0],key[1],v[0][0],v[0][1]),x))
                xwts.append(xwt)


            g = create_gaintable_from_visibility_para(viss3[key][0][1], 3)
            solve_from_X_para(xs, xwts, g, npol=viss3[key][0][1].npol)
            gts.append((key, g))

        gaintable_right(gt, gts)






    def test_applay_gaintable(self):
        '''
            测试apply_gaintable
        :return:
        '''
        pass

    # def pass_test(self):
    #     pass





def create_new_wcs_new_shape(wcs, shape):
    newwcs = copy.deepcopy(wcs)
    newwcs.wcs.cdelt[0] = -0.001 * 180 / np.pi
    newwcs.wcs.cdelt[1] = 0.001 * 180 / np.pi
    newshape = np.array(shape)
    newshape[-2] /= 2
    newshape[-1] /= 2
    return newwcs, newshape


if __name__ == '__main__':
    unittest.main()

