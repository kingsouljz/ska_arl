from arl.util.testing_support import *
from astropy.coordinates import SkyCoord
from arl.skycomponent.operations import *
from arl.data.polarisation import *
from arl.data.data_models import *
from arl.visibility.base import *
from astropy import constants
from arl.image.operations import *
from reproject import reproject_interp
from arl.skycomponent.operations import insert_skycomponent
from Constants import *
import math
import numpy as np

class visibility_for_para:
    '''
        被并行化后的visibility_buffer的数据结构
        有三种模式，pol模式按照 pol轴对vis进行了拆分 npol模式对pol轴进行了保留， chan模式则将chan序号相同的visibility存储在同一个类中
        因此pol和npol模式的nvis默认为1， chan模式的nvis为 time * base数
    '''
    #TODO 暂时将phasecentre等telescope数据也存储在该类中, 视之后情况而定，因此同一个telescope数据被复制nvis份
    def __init__(self, vis, uvw, time, frequency, bandwidth, integration_time, antenna1, antenna2,
                 weight, imaging_weight, visibility, mode, nvis=1):
        '''
        以下分别针对三种模式       pol      npol    frequency
        :param vis: 最重要的信息 [nvis, 1] [nvis, 4] [nvis, 4]
        :param uvw: 对应的uvw数据，其中隐含了time和baseline的信息，具体可以参照arl.visibility.basecreate_visibility方法中uvw生成的过程
                               [nvis, 3] [nvis, 3] [nvis, 3]
        :param time: 时间角数据  [nvis, 1] [nvis, 1] [nvis, 1]
        :param frequency: 对应的频率 type同上
        :param bandwidth: 对应的带宽 type同上
        :param integration_time:       type同上
        :param antenna1: baseline的第一个天线id type同上
        :param antenna2: baseline的第二个天线id type同上
        :param weight:   vis的在总的visibility中的权重 type同vis
        :param imaging_weight: type同vis
        :param visibility: 所有的visibility_for_para均切分自该实例，此处传入提供一些公共信息如phasecentre,configuration等，
            将来可以考虑将其作为广播数据传给所有节点，而不是为每个visibility_for_para复制一份该信息
        :param mode: 模式，String
        :param nvis: data的0维的长度
        '''
        npol = visibility.polarisation_frame.npol
        if mode == "pol":
            desc = [('uvw', '>f8', (3,)),
                    ('time', '>f8'),
                    ('frequency', '>f8'),
                    ('channel_bandwidth', '>f8'),
                    ('integration_time', '>f8'),
                    ('antenna1', '>i8'),
                    ('antenna2', '>i8'),
                    ('vis', '>c16'),
                    ('weight', '>f8'),
                    ('imaging_weight', '>f8'),
                    ]
        elif mode == "npol":
            desc = [('uvw', '>f8', (3,)),
                    ('time', '>f8'),
                    ('frequency', '>f8'),
                    ('channel_bandwidth', '>f8'),
                    ('integration_time', '>f8'),
                    ('antenna1', '>i8'),
                    ('antenna2', '>i8'),
                    ('vis', '>c16', (npol,)),
                    ('weight', '>f8', (npol,)),
                    ('imaging_weight', '>f8', (npol,)),
                    ]
        elif mode == "chan":
            desc = [('uvw', '>f8', (3,)),
                    ('time', '>f8'),
                    ('frequency', '>f8'),
                    ('channel_bandwidth', '>f8'),
                    ('integration_time', '>f8'),
                    ('antenna1', '>i8'),
                    ('antenna2', '>i8'),
                    ('vis', '>c16', (npol,)),
                    ('weight', '>f8', (npol,)),
                    ('imaging_weight', '>f8', (npol,)),
                    ]
        data = np.zeros(shape=nvis, dtype=desc)
        data['uvw'] = uvw
        data['time'] = time
        data['frequency'] = frequency
        data['channel_bandwidth'] = bandwidth
        data['integration_time'] = integration_time
        data['antenna1'] = antenna1
        data['antenna2'] = antenna2
        data['vis'] = vis
        data['weight'] = weight
        data['imaging_weight'] = imaging_weight
        self.nvis = nvis
        self.mode = mode
        self.data = data
        self.npol = npol
        # TODO 下面的信息可以考虑广播，减少占用空间
        self.phasecentre = visibility.phasecentre
        self.polarisation_frame = visibility.polarisation_frame
        # self.configuration = visibility.configuration

    @property
    def block_uvw(self):
        '''
            拿到blockvisibilit中的uvw值, 做一个简单的变换即可
        :return:
        '''
        if self.mode == "npol" or self.mode == "pol":
            return data['uvw'][0] * constants.c.value / data['frequency'][0]
        else:
            return data['uvw'] * constants.c.value / data['frequency'][0]

    def uvw(self):
        if self.mode == "npol" or self.mode == "pol":
            return data['uvw'][0]
        else:
            return data['uvw']

    def vis(self):
        if self.mode == "npol" or self.mode == "pol":
            return data['vis'][0]
        else:
            return data['uvw']

    def __str__(self):
        ret = "mode: %s, nvis: %s, npol: %s \n" %(self.mode, self.nvis, self.npol)
        ret += "vis " + str(self.data['vis']) + "\n"
        return ret


class visibility_share:
    '''
        被并行化后的visibility的共享数据
        phasecentre
        polarisation_frame
        configuration
        cindex
    '''
    # TODO 在并行化程序中，blockvis没有意义
    # TODO cindex 目前还不知道其与vis关系，可以考虑对该类实例进行广播，并去掉之前visibility_for_para中的同名数据，减小空间占用
    def __init__(self, visibility: Visibility):
        self.phasecentre = visibility.phasecentre
        self.polarisation_frame = visibility.polarisation_frame
        self.configuration = visibility.configuration
        self.cindex = visibility.cindex
        self.npol = visibility.polarisation_frame.npol
        self.nvis = visibility.nvis

    def __str__(self):
        ret = "phasecentre: %s, polarisation_frame_type: %s, configuration: %s" % (str(self.phasecentre),
        str(self.polarisation_frame), str(self.configuration))

class block_visibility_for_para:
    '''
        coalesce之后的visibility
    '''
    pass

class gaintable_for_para:
    """
        并行化的gaintable
    """
    # TODO 初步确定的结构
    def __init__(self, ):
        pass

class image_for_para:
    '''
        image按照nchanl和npol切开后的数据结构便于并行化
        每一个image由frequency time facet polarisation唯一决定
        另外不同的切片会有不同的wcs, wcs.wcs.crpix值后产生变化，其他的数据不变
        每个被切片的image产生的第一个切片的wcs和原image的wcs是相同的
    '''
    # TODO wcs的具体结构, 其中除了wcs、data和facet将来可以考虑去掉其他项，因为它们会被保存在rdd的键值对的键中
    def __init__(self, data, wcs:int, frequency:int, time:int, facet:int, polarisation:int):
        '''

        :param data: 主要的数据， [ny, nx]
        :param wcs:  wcs,切片后几乎每个个切片的wcs都和未切片的image的wcs不相同   WCS
        :param frequency: chan编号 int
        :param time: time编号 int
        :param facet: 切片编号 int
        :param polarisation: npol编号 int
        '''
        self.time = time
        self.polarisation = polarisation
        self.frequency = frequency
        self.facet = facet
        self.data = data
        self.wcs = wcs
    @property
    def ny(self): return self.data.shape[0]

    @property
    def nx(self): return self.data.shape[1]

    @property
    def shape(self):
        return self.data.shape

    @property
    def phasecentre(self):
        return SkyCoord(self.wcs.wcs.crval[0] * u.deg, self.wcs.wcs.crval[1] * u.deg)

    def __str__(self):
        ret = "frequency: " + str(self.frequency) + " time: " + str(self.time) + " facet " + str(self.facet)\
        + " polarisation " + str(self.polarisation) + "\n"
        ret += "shape: " + str(self.shape) + "\n"
        # ret += "data: " + str(self.data)
        return ret


class image_share:
    '''
        被并行化后的image的共享数据
        wcs
        phasecentre
        frequency
        polarisation_frame
    '''
    # TODO 存储所有image的共享数据， 可以广播给每个节点节约空间占用
    def __init__(self, frequency, polarisation_frame: PolarisationFrame, wcs):
        # 每个chanel对应的frequecy实际的值
        self.frequency = frequency
        self.polarisation_frame = polarisation_frame
        self.wcs = wcs

    @property
    def shape(self):
        return (self.frequency.shape[0], self.polarisation_frame.npol)


# ===工具类函数===
def float_equal(a, b):
    '''
        判断两个float数是否相等，差小于1e-7即认为相等
    :param a:
    :param b:
    :return:
    '''
    # print(math.fabs(a - b))
    if math.fabs(a - b) < math.pow(10, -PRECISION):
        return True
    else:
        return False

def uvw_equal(a, b):
    '''
        判断两个uvw是否相等
    :param a:
    :param b:
    :return:
    '''
    for x,y in zip(a, b):
        if math.fabs(x - y) >= math.pow(10, -PRECISION):
            return False

    return True



def complex_equal(a: np.complex_, b: np.complex_):
    '''
        判断两个complex数是否相等，虚部和实部之差均小于1e-7即认为相等
    :param a:
    :param b:
    :return:
    '''
    if (math.fabs(a.imag - b.imag) < math.pow(10, -PRECISION)) and (math.fabs(a.real - b.real) < math.pow(10, -PRECISION)):
        return True
    else:
        return False

def wcs4_to_wcs2(wcs4: WCS) -> WCS:
    '''
        将wcs4转换为wcs2, 主要是将四维的Image 拆分成二维的image_for_para时使用
    :param wcs4: axis为4的wcs
    :return: axis为2的wcs
    '''
    newwcs = WCS(naxis=2)
    newwcs.wcs.crpix = wcs4.wcs.crpix[0:2]
    newwcs.wcs.cdelt = wcs4.wcs.cdelt[0:2]
    newwcs.wcs.crval = wcs4.wcs.crval[0:2]
    newwcs.wcs.ctype = [wcs4.wcs.ctype[0], wcs4.wcs.ctype[1]]
    newwcs.wcs.radesys = wcs4.wcs.radesys
    newwcs.wcs.equinox = wcs4.wcs.equinox
    return newwcs

def create_image_from_array_para(im , y1:int, y2:int, x1:int, x2:int, id:int, frequency, time, polarisation, wcs: WCS = None) -> image_for_para:
    '''
    以im二维数组作为data的内容，并按照一定坐标切分，创造新的image_para类
    :param im:  被切分的image， 类型[y, x]
    :param y1:  切分的坐标：[y1 ~ y2, x1 ~ x2]
    :param y2:
    :param x1:
    :param x2:
    :param id:  赋予每个facet的id
   :param wcs: 每个facet的wcs
    :return: 切分后的image
    '''
    fim = image_for_para(im[y1:y2, x1:x2], None, frequency, time, id, polarisation)
    if wcs is None:
        fim.wcs = None
    else:
        fim.wcs = wcs.deepcopy()
    assert type(fim) == image_for_para, "Type is %s" % type(fim)
    return fim

def image_make_facets(im: image_for_para, facets: int, polarisation_frame: PolarisationFrame):
    '''
        将image切分为facets * facets个小image的生成器
    :param im: 被切片的image
    :param facets: 切片片数，type:int
    :param polarisation_frame: 极化框架，type:PolarisationFrame, 和被切片前的一样，由image_share提供
    :return: 被前片后的image_for_para list
    '''
    # 每个切片的长和宽
    dx = int(im.nx // facets)
    dy = int(im.ny // facets)
    # 每个切片的id
    id = 0
    for y in range(0, im.ny, dy):
        for x in range(0, im.nx, dx):
            # 调整切片后每个image的wcs
            wcs = im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y
            # 作为生成器返回切片后的image
            yield create_image_from_array_para(im.data, y, y + dy, x, x + dx, id, im.frequency, im.time, im.polarisation, wcs)
            id += 1

def image_to_image_para(im: Image, facet: int):
    '''
        将image按照chan和pol轴拆分为并行image_para, 并切片
    :param im: 待切分和切片的原Image
    :return: ims和img_share, 类型: [nchan * npol * facet] list[image_for_para]
    '''
    image_graph = copy.deepcopy(im) # 创造副本，非原地函数
    ims = []
    img_share = image_share(image_graph.frequency, image_graph.polarisation_frame, image_graph.wcs)
    for i in range(image_graph.nchan):
        for j in range(image_graph.npol):
            # 按照chan和pol轴拆分为并行image_para
            temp_im = image_for_para(image_graph.data[i, j], wcs4_to_wcs2(image_graph.wcs), i, 0, 0, j)
            # 上一步生成的并行image_para进一步切片
            for k in image_make_facets(temp_im, facet, img_share.polarisation_frame, ):
                ims.append(k)
    return ims, img_share

def image_restore(data, facets: int):
    '''
        image_gather调用的一个辅助子函数
    :param data:
    :param facets:
    :return:
    '''
    ny, nx = data.shape
    dy = int(ny // facets)
    dx = int(nx // facets)
    for y in range(0, ny, dy):
        for x in range(0, nx, dx):
            yield data[y:y+dy, x:x+dx]

def image_gather(ims: list, facets: int, wcs: WCS, output) -> image_for_para:
    '''
    将被切片的image_for_para聚集成一个大的image_for_para， image_make_facets的逆函数
    :param ims: 所有来同一个image的切片image的集合
    :param facets: 切片的个数
    :param wcs: 原本的wcs
    :param output: 与原image shape相同的0数组
    :return: 还原后的image
    '''
    im = image_for_para(None, wcs, ims[0].frequency, ims[0].time, 0, ims[0].polarisation)
    for i, facet in enumerate(image_restore(output, facets)):
        facet[...] = ims[i].data[...]
    im.data = output
    return im

def image_para_to_image(ims: list, image_share: image_share) -> Image:
    '''
        将并行image_for_para还原为原本的Image类，验证算法正确性用， image_to_image_para的逆函数
    :param ims:  只有一个facet的并行Image list 类型：list[image_para]
    :param image_share: 并行Image的共享消息
    :return: 还原后的image
    '''
    im = Image()
    data = np.zeros([image_share.shape[0], image_share.shape[1], ims[0].ny, ims[0].nx], dtype=ims[0].data.dtype)
    for i in ims:
        data[i.frequency][i.polarisation] = i.data
    im.data = data
    im.wcs = ims[0].wcs
    im.polarisation_frame = image_share.polarisation_frame
    return im

def image_para_facet_to_image(ims: list, facets: int, image_share: image_share) -> Image:
    nchan, npol = image_share.shape
    piece = facets * facets
    im = []
    for i in range(nchan):
        for j in range(npol):
            temp = ims[i * npol * piece + j * piece: i * piece * npol + j * piece + piece]
            im.append(image_gather(temp, facets, ims[i * piece * npol + j * piece].wcs,
                                     np.zeros([ims[0].data.shape[1] * facets, ims[0].data.shape[0] * facets],
                                              dtype=ims[0].data.dtype)))
    return image_para_to_image(im, image_share)

def image_right(a:Image, b:Image):
    '''
    验证两个image是否完全相等
    :param a:
    :param b:
    :return:
    '''
    assert a.shape == b.shape, "two Images' shape are different %s and %s" % (a.shape, b.shape)
    ncha, npol, ny, nx = a.shape
    for i in range(ncha):
        for j in range(npol):
            for k in range(ny):
                for l in range(nx):
                    # 浮点数问题，用round保留十位，便于比较
                    assert type(a.data[i,j,k,l]) == type(b.data[i,j,k,l]), "two images' data type are different %s and %s"%(type(a.data[i,j,k,l]), type(b.data[i,j,k,l]))

                    if type(a.data[i,j,k,l]) == np.float_ or type(a.data[i,j,k,l]) == np.int_:
                        assert float_equal(a.data[i,j,k,l], b.data[i,j,k,l]), "two Images' data are different: %s and %s" %(a.data[i,j,k,l],b.data[i,j,k,l])
                        # print(a.data[i,j,k,l], b.data[i,j,k,l])

                    elif type(a.data[i,j,k,l]) == np.complex_:
                        assert complex_equal(a.data[i,j,k,l], b.data[i,j,k,l]), "two Images' data are different: %s and %s" %(a.data[i,j,k,l], b.data[i,j,k,l])
                    # TODO wcs的验证方法
    assert a.polarisation_frame.type == b.polarisation_frame.type, "two Images' polarisation_frame are different %s and %s" % (a.polarisation_frame.type, b.polarisation_frame.type)

    print("pass the test!")

def visibility_to_visibility_para(vis: Visibility, mode):
    '''
        由visibility生成visibility_for_para
        mode: pol 按照 (fre,time,ant1,ant2,pol) 进行划分
              npol 按照 (fre, time, an1, ant2) 进行划分
              frequency 按照 (fre, id) 进行划分
    :param visibility: 待拆分的Visibiliy
    :param mode:
    :return: 并行化的visibility_for_para 键值对(id, vis) 和 visibility的共享数据
    '''
    vis = copy.deepcopy(vis)
    viss = []
    # 将不同的frequency 和 time投影到唯一的从0开始的整数上便于作为key, # antenna1 和 antenna2的组合还未找到一个较好的投影方法投影到baseline上去
    ufrequency = numpy.unique(vis.data['frequency'])
    m = {}
    for idx, chan in enumerate(ufrequency):
        m[chan] = idx

    utime = numpy.unique(vis.data['time'])
    t = {}
    for idx, time in enumerate(utime):
        t[time] = idx

    if mode == 'pol':
        for i in range(vis.nvis):
            for j in range(vis.polarisation_frame.npol):
                v = visibility_for_para(vis.data['vis'][i][j], vis.data['uvw'][i]
                               ,vis.data['time'][i],vis.data['frequency'][i],vis.data["channel_bandwidth"][i]
                               ,vis.data['integration_time'][i],vis.data['antenna1'][i],vis.data['antenna2'][i]
                               ,vis.data['weight'][i][j], vis.data["imaging_weight"][i][j], vis, mode)

                viss.append(((m[v.data['frequency'][0]], t[v.data['time'][0]], v.data['antenna1'][0], v.data['antenna2'][0], j), v))
                # 键值对格式：((nchan, time, ant1, ant2, npol), v)

    elif mode == 'npol':
        for i in range(vis.nvis):
            v = visibility_for_para(vis.data['vis'][i], vis.data['uvw'][i]
                                      ,vis.data['time'][i],vis.data['frequency'][i],vis.data["channel_bandwidth"][i]
                               ,vis.data['integration_time'][i],vis.data['antenna1'][i],vis.data['antenna2'][i]
                               ,vis.data['weight'][i], vis.data["imaging_weight"][i], vis, mode)

            viss.append(((m[v.data['frequency'][0]], t[v.data['time'][0]], v.data['antenna1'][0], v.data['antenna2'][0]), v))
            # 键值对格式：((nchan, time, ant1, ant2), v)

    elif mode == 'chan':
        for i in range(NCHAN):
            nvis = vis.data.shape[0] // NCHAN
            actual_frequency = i
            v = visibility_for_para(vis.data['vis'][actual_frequency::5], vis.data['uvw'][actual_frequency::5]
                                 , vis.data['time'][actual_frequency::5], vis.data['frequency'][actual_frequency::5],
                                 vis.data["channel_bandwidth"][actual_frequency::5]
                                 , vis.data['integration_time'][actual_frequency::5], vis.data['antenna1'][actual_frequency::5],
                                 vis.data['antenna2'][actual_frequency::5]
                                 , vis.data['weight'][actual_frequency::5], vis.data["imaging_weight"][actual_frequency::5], vis, mode, nvis)

            viss.append(((m[v.data['frequency'][0]]), v))
            # 键值对格式：((nchan), v)
    vis_share = visibility_share(vis)
    return viss, vis_share

def visibility_para_to_visibility(viss, mode, visibility: visibility_share) -> Visibility:
    '''
        拆分后的visibility还原为原visibility
    :param viss: 待还原的visbility_for_para list
    :param mode:
    :param visibility: 共享的visibility数据
    :return:
    '''
    npol = visibility.npol
    nvis = visibility.nvis
    desc = [('uvw', '>f8', (3,)),
            ('time', '>f8'),
            ('frequency', '>f8'),
            ('channel_bandwidth', '>f8'),
            ('integration_time', '>f8'),
            ('antenna1', '>i8'),
            ('antenna2', '>i8'),
            ('vis', '>c16', (npol,)),
            ('weight', '>f8', (npol,)),
            ('imaging_weight', '>f8', (npol,))]
    data = numpy.zeros(shape=[nvis], dtype=desc)
    if mode == 'pol':
        for i in range(nvis):
            for j in range(npol):
                id = i * npol + j
                data['vis'][i][j] = viss[id][1].data['vis']
                data['weight'][i][j] = viss[id][1].data['weight']
                data['imaging_weight'][i][j] = viss[id][1].data['imaging_weight']
            data['uvw'][i] = viss[i * npol][1].data['uvw']
            data['time'][i] = viss[i * npol][1].data['time']
            data['frequency'][i] = viss[i * npol][1].data['frequency']
            data['channel_bandwidth'][i] = viss[i * npol][1].data['channel_bandwidth']
            data['integration_time'][i] = viss[i * npol][1].data['integration_time']
            data['antenna1'][i] = viss[i * npol][1].data['antenna1']
            data['antenna2'][i] = viss[i * npol][1].data['antenna2']

    elif mode == 'npol':
        for i in range(visibility.nvis):
            data['uvw'][i] = viss[i][1].data['uvw']
            data['time'][i] = viss[i][1].data['time']
            data['frequency'][i] = viss[i][1].data['frequency']
            data['channel_bandwidth'][i] = viss[i][1].data['channel_bandwidth']
            data['integration_time'][i] = viss[i][1].data['integration_time']
            data['antenna1'][i] = viss[i][1].data['antenna1']
            data['antenna2'][i] = viss[i][1].data['antenna2']
            data['vis'][i] = viss[i][1].data['vis']
            data['weight'][i] = viss[i][1].data['weight']
            data['imaging_weight'][i] = viss[i][1].data['imaging_weight']

    elif mode == 'chan':
        nchan = len(viss)
        for i in range(nchan):
            data['uvw'][i::5] = viss[i][1].data['uvw']
            data['time'][i::5] = viss[i][1].data['time']
            data['frequency'][i::5] = viss[i][1].data['frequency']
            data['channel_bandwidth'][i::5] = viss[i][1].data['channel_bandwidth']
            data['integration_time'][i::5] = viss[i][1].data['integration_time']
            data['antenna1'][i::5] = viss[i][1].data['antenna1']
            data['antenna2'][i::5] = viss[i][1].data['antenna2']
            data['vis'][i::5] = viss[i][1].data['vis']
            data['weight'][i::5] = viss[i][1].data['weight']
            data['imaging_weight'][i::5] = viss[i][1].data['imaging_weight']

    ret = Visibility(data=data, cindex=visibility.cindex, phasecentre=visibility.phasecentre,
                     configuration=visibility.configuration, polarisation_frame=visibility.polarisation_frame)

    return ret

def visibility_right(a: Visibility, b: Visibility):
    '''
        验证两个Visibility是否完全相等
    :param a:
    :param b:
    :return:
    '''
    # TODO cindex,blockvis,phasecentre,configuration验证方法待完成
    assert a.nvis == b.nvis, "two Visibilities' shape are different: %s and %s" %(a.nvis, b.nvis)
    assert a.polarisation_frame.type == b.polarisation_frame.type, "two Visibilities' polarisation_frame are different"
    for i in range(a.nvis):
        assert uvw_equal(a.data['uvw'][i], b.data['uvw'][i]), "uvw are different %s and %s" % (a.data['uvw'][i], b.data['uvw'][i])
        assert a.data['time'][i] == b.data['time'][i], "time are different"
        assert a.data['frequency'][i] == b.data['frequency'][i], "frequency are different %s and %s"% (a.data['frequency'][i], b.data['frequency'][i])
        assert a.data['channel_bandwidth'][i] == b.data['channel_bandwidth'][i], "channel_bandwidth are different"
        assert a.data['integration_time'][i] == b.data['integration_time'][i], "integration_time are different"
        assert a.data['antenna1'][i] == b.data['antenna1'][i], "antenna1 are different"
        assert a.data['antenna2'][i] == b.data['antenna2'][i], "antenna2 are different"
        if type(a.data['vis'][i]) == type(b.data['vis'][i]) and type(a.data['vis'][i] == np.complex_):
            for j in range(a.polarisation_frame.npol):
                assert complex_equal(a.data['vis'][i][j], b.data['vis'][i][j]), "vis are different %s and %s" % (a.data['vis'][i][j], b.data['vis'][i][j])
        assert (a.data['weight'][i] == b.data['weight'][i]).all(), "weight are different"
        assert (a.data['imaging_weight'][i] == b.data['imaging_weight'][i]).all(), "imaging_weight are different"

    print("pass the test!")


# ===实际并行化后的arl函数===
# =============# =============# reppre_ifft =============# =============# =============
def insert_skycomponent_para(im: image_for_para, sc: Union[Skycomponent, List[Skycomponent]], insert_method='',
                        bandwidth=1.0, support=8) -> image_for_para:
    '''
        将skycomponent中的信息插入到image中并生成一个新的image，该步骤可以并行化处理，不用修改内部逻辑
        只会对image_for_para的data产生影响，并且插入的值和image_for_para的chan、pol有关，插入的data的位置和image_for_para的wcs有关
    :param im: 被插入的image
    :param sc: 插入的skycomponent，可以有多个skycomponent
    :param insert_method: 插入方法，四种分别为: Lanczos Sinc PSWF 和 缺省方法
    :param bandwidth:
    :param support:
    :return: 新的image
    '''
    assert type(im) == image_for_para

    support = int(support / bandwidth)

    ny, nx = im.data.shape

    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    for comp in sc:
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        pixloc = skycoord_to_pixel(comp.direction, im.wcs, 1, 'wcs')
        if insert_method == "Lanczos":
            insert_array_para(im.data, pixloc[0], pixloc[1], comp.flux[im.frequency, im.polarisation], bandwidth, support,
                         insert_function=insert_function_L)
        elif insert_method == "Sinc":
            insert_array_para(im.data, pixloc[0], pixloc[1], comp.flux[im.frequency, im.polarisation], bandwidth, support,
                         insert_function=insert_function_sinc)
        elif insert_method == "PSWF":
            insert_array_para(im.data, pixloc[0], pixloc[1], comp.flux[im.frequency, im.polarisation], bandwidth, support,
                         insert_function=insert_function_pswf)
        else:
            y, x = numpy.round(pixloc[1]).astype('int'), numpy.round(pixloc[0]).astype('int')
            if x >= 0 and x < nx and y >= 0 and y < ny:
                im.data[y, x] += comp.flux[im.frequency, im.polarisation]

    return im

# 以下三个函数不受并行化影响
def insert_function_sinc(x):
    s = numpy.zeros_like(x)
    s[x != 0.0] = numpy.sin(numpy.pi*x[x != 0.0])/(numpy.pi*x[x != 0.0])
    return s

def insert_function_L(x, a = 5):
    L = insert_function_sinc(x) * insert_function_sinc(x/a)
    return L

def insert_function_pswf(x, a=5):
    from arl.fourier_transforms.convolutional_gridding import grdsf
    return grdsf(abs(x)/a)[0]

def insert_array_para(im: image_for_para, x, y, flux, bandwidth=1.0, support=7, insert_function=insert_function_sinc):
    '''
        并行化版本，和缺省方式不同，根据insert_function的不同的插入的大小和值也不同
    :param im: image
    :param x: pixloc，插入的位置相关
    :param y:
    :param flux: 插入的值
    :param bandwidth:
    :param support:
    :param insert_function: 插入的方式
    :return:
    '''
    ny, nx = im.shape
    intx = int(numpy.round(x))
    inty = int(numpy.round(y))
    fracx = x - intx
    fracy = y - inty
    gridx = numpy.arange(-support, support)
    gridy = numpy.arange(-support, support)

    insert = numpy.outer(insert_function(bandwidth * (gridy - fracy)),
                         insert_function(bandwidth * (gridx - fracx))) # [support * 2, support * 2]

    insertsum = numpy.sum(insert)
    assert insertsum > 0, "Sum of interpolation coefficients %g" % insertsum
    insert = insert / insertsum # [support * 2, support * 2]

    for iy in gridy:
        for ix in gridx:
            if (iy + inty >= 0 and iy + inty < ny and ix + intx >=0 and ix + intx < nx):
                im[iy + inty, ix + intx] += flux * insert[iy + support, ix + support]

    return im

def fft(a):
    """ Fourier transformation from image to grid space

    .. note::

        If there are four axes nthen the last outer axes are not transformed

    :param a: image in `lm` coordinate space
    :return: `uv` grid
    """
    # TODO 傅里叶变换切片后不等效，但是能对不同nchan和npol上的image并行处理
    if (len(a.shape) == 4):
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a, axes=[2, 3])), axes=[2, 3])
    else:
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))

def reproject_image_para(im: image_for_para, newwcs: WCS, shape=None) -> (Image, Image):
    '''
        按照一个新的WCS将image_for_para重新投影到shape的大小
    :param im: 原image
    :param newwcs: 新的wcs
    :param shape: 投影的大小
    :return: 一个新的image
    '''
    # TODO 切片后，投影不等效，能对不同nchan和npol上的image并行处理
    assert type(im) == image_for_para
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=True)
    return create_image_from_array_para(rep, 0, rep.shape[0], 0, rep.shape[1], im.facet, im.frequency, im.time, im.polarisation, newwcs)\
        , create_image_from_array_para(foot, 0, foot.shape[0], 0, foot.shape[1], im.facet, im.frequency, im.time, im.polarisation,  newwcs)

# =============# =============# degrid =============# =============# =============
def predict_2d_base_para(vis: visibility_for_para, model: image_for_para, **kwargs) -> visibility_for_para:
    '''
        用image 预测 visibility 生成新的visibility
    :param vis: 待预测的visibility
    :param model:
    :param kwargs:
    :return: 预测后的visiblity
    '''
    ny, nx = model.data.shape
    uvw_mode, shape, padding, vuvwmap = get_uvw_map_para(vis, model)
    kernel_name, gcf, vkernellist = get_kernel_list_para(vis, model, **kwargs)
    # 此处gcf大小和padding后的model大小相同, 然后做fft，fft和padding函数均不用变动
    uvgrid = fft((pad_mid(model.data, int(round(padding * nx))) * gcf).astype(dtype=complex))
    vis.data['vis'] = convolutional_degrid_para(vkernellist, vis.data['vis'].shape, uvgrid,
                                            vuvwmap, model)
    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image_para(vis, model, tangent=True, inverse=True)
    return svis

def shift_vis_to_image_para(vis: visibility_for_para, im: image_for_para, tangent: bool = True, inverse: bool = False) -> visibility_for_para:
    '''
        旋转visibility的phasecentre到image的phasecentre，改变vis的uvw等值
    :param vis:
    :param im:
    :param tangent:
    :param inverse:
    :return:
    '''
    ny, nx = im.data.shape
    image_phasecentre = pixel_to_skycoord(nx // 2, ny // 2, im.wcs, origin=1)

    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        vis = phaserotate_visibility_para(vis, image_phasecentre, tangent=tangent, inverse=inverse)
        vis.phasecentre = im.phasecentre


    return vis

def convolutional_degrid_para(kernel_list, vshape, uvgrid, vuvwmap, im: image_for_para):
    '''

    :param kernel_list:
    :param vshape:
    :param uvgrid:
    :param vuvwmap:
    :param im:
    :return:
    '''
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    ny, nx = uvgrid.shape
    vis = numpy.zeros(vshape, dtype='complex')

    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2 # (1,1)

    pol = im.polarisation
    # TODO 此处去掉了 len(kernels) > 1的分支， 因为本阶段不会触发该条件
    coords = (im.frequency, x[0], y[0], xf[0], yf[0])
    ckernel0 = numpy.conjugate(kernels[0]) # (8,8,8,8)
    # vis的time和base由uvw消解掉
    vis[0, pol] = numpy.sum(uvgrid[coords[2]:coords[2]+gh, coords[1]:coords[1]+gw] * ckernel0[coords[4], coords[3], :, :])
    return vis

# TODO 不用改动
def frac_coord(npixel, kernel_oversampling, p):
    """ Compute whole and fractional parts of coordinates, rounded to
    """
    assert numpy.array(p >= -0.5).all() and numpy.array(
        p < 0.5).all(), "Cellsize is too large: uv overflows grid uv= %s" % str(p)
    x = npixel // 2 + p * npixel
    flx = numpy.floor(x + 0.5 / kernel_oversampling)
    fracx = numpy.around((x - flx) * kernel_oversampling)
    return flx.astype(int), fracx.astype(int)

def get_kernel_list_para(vis: visibility_for_para, im: image_for_para, **kwargs):
    """
        暂时去掉了W-Projection if分支，本阶段用不到该分支
        个人认为 padding对应facet
    """

    shape = im.data.shape
    npixel = shape[1]
    cellsize = numpy.pi * im.wcs.wcs.cdelt[1] / 180.0

    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    padding = get_parameter(kwargs, "padding", FACETS)
    support = get_parameter(kwargs, "support", 3)

    gcf, _ = anti_aliasing_calculate((padding * npixel, padding * npixel), oversampling)

    kernelname = '2d'
    kernel_list = (0, [anti_aliasing_calculate((padding * npixel, padding * npixel), oversampling, support)[1]])

    return kernelname, gcf, kernel_list

# TODO 此函数不变化
def anti_aliasing_calculate(shape, oversampling=1, support=3):
    """
    """

    # 2D Prolate spheroidal angular function is separable
    ny, nx = shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d, _ = grdsf(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]

    s1d = 2 * support + 2
    nu = numpy.arange(-support, +support, 1.0 / oversampling)
    kernel1d = grdsf(nu / support)[1]
    l1d = len(kernel1d)
    # Rearrange to get the convolution function isolated by (yf, xf). For this convolution function
    # the result is heavily redundant but it does fit well into the general framework
    kernel4d = numpy.zeros((oversampling, oversampling, s1d, s1d))
    for yf in range(oversampling):
        my = range(yf, l1d, oversampling)[::-1]
        for xf in range(oversampling):
            mx = range(xf, l1d, oversampling)[::-1]
            kernel4d[yf, xf, 2:, 2:] = numpy.outer(kernel1d[my], kernel1d[mx])
    return gcf, (kernel4d / numpy.sum(kernel4d[0, 0, :, :])).astype('complex')

# TODO 此函数不变化
def coordinates(npixel: int) -> object:
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2

    """
    return (numpy.arange(npixel) - npixel // 2) / npixel

# TODO 此函数不变化
def grdsf(nu):
    """Calculate PSWF using an old SDE routine re-written in Python
    """
    p = numpy.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                     [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                     [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    _, np = p.shape
    _, nq = q.shape

    nu = numpy.abs(nu)

    nuend = numpy.zeros_like(nu)
    part = numpy.zeros(len(nu), dtype='int')
    part[(nu >= 0.0) & (nu < 0.75)] = 0
    part[(nu > 0.75) & (nu < 1.0)] = 1
    nuend[(nu >= 0.0) & (nu <= 0.75)] = 0.75
    nuend[(nu > 0.75) & (nu < 1.0)] = 1.0

    delnusq = nu ** 2 - nuend ** 2

    top = p[part, 0]
    for k in range(1, np):
        top += p[part, k] * numpy.power(delnusq, k)

    bot = q[part, 0]
    for k in range(1, nq):
        bot += q[part, k] * numpy.power(delnusq, k)

    grdsf = numpy.zeros_like(nu)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = numpy.abs(nu > 1.0)
    grdsf[ok] = 0.0

    # Return the gridding function and the grid correction function
    return grdsf, (1 - nu ** 2) * grdsf

# TODO 此函数不变化
def pad_mid(ff, npixel):
    """
    """
    if len(ff.shape) == 2:
        ny, nx = ff.shape
        cx = nx // 2
        cy = ny // 2
        if npixel == nx:
            return ff
        assert npixel > nx == ny
        pw = ((npixel // 2 - cy, npixel // 2 - cy),
              (npixel // 2 - cx, npixel // 2 - cx))
        return numpy.pad(ff,
                         pad_width=pw,
                         mode='constant',)

def get_uvw_map_para(vis: visibility_for_para, im: image_for_para, padding=FACETS):
    ny, nx = im.data.shape
    shape = (1, int(round(padding * ny)), int(round(padding * nx)))
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    vuvwmap = uvwscale * vis.data['uvw']
    uvw_mode = "2d"

    return uvw_mode, shape, padding, vuvwmap

def predict_facets_para(vis: visibility_for_para, model: image_for_para, predict_function=predict_2d_base_para, **kwargs) -> Visibility:
    '''
        predict的主入口
    :param vis:
    :param model:
    :param predict_function:
    :param kwargs:
    :return:
    '''
    # TODO 只改变了visibility类的vis值，其他的configuration都没有发生改变, 并且随着facet的数量不同得到的结果也不同, 具体并行操作需要进一步探究
    return predict_with_image_iterator_para(vis, model, predict_function=predict_function,
                                       **kwargs)

def predict_with_image_iterator_para(vis: visibility_for_para, model: image_for_para,
                                predict_function=predict_2d_base_para, **kwargs) -> visibility_for_para:
    '''
    :param vis:
    :param model:
    :param predict_function:
    :param kwargs:
    :return:
    '''
    # TODO 原本此步应该是对所有切片的叠加，怎样并行？
    result = copy.deepcopy(vis)
    result.data['vis'][...] = 0.0
    result = predict_function(result, model, **kwargs)
    ret = copy.deepcopy(vis)
    ret.data['vis'][...] += result.data['vis'][...]
    return ret

# =============# =============# pharotpre_dft_sumvis_ =============# =============# =============
def decoalesce_visibility_para(vis: visibility_for_para):
    '''

    :param vis:
    :return:
    '''
    # TODO 暂时不需要此方法，因为不明确coalesce的方法


def phaserotate_visibility_para(vis: visibility_for_para, newphasecentre: SkyCoord, tangent=True, inverse=False) -> visibility_for_para:
    """
        并行版
    :param vis: Visibility to be rotated
    :param newphasecentre:
    :param tangent: Stay on the same tangent plane? (True)
    :param inverse: Actually do the opposite
    :return: Visibility
    """
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)
    # No significant change?
    if numpy.abs(n) > 1e-15:

        # Make a new copy
        newvis = copy_visibility(vis)

        phasor = simulate_point(newvis.data['uvw'], l, m)

        if inverse:
            newvis.data['vis'][0] *= phasor
        else:
            newvis.data['vis'][0] *= numpy.conj(phasor)

        if not tangent:
            if inverse:
                xyz = uvw_to_xyz(vis.data['uvw'], ha=-newvis.phasecentre.ra.rad, dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad)[...]
            else:
                # This is the original (non-inverse) code
                xyz = uvw_to_xyz(newvis.data['uvw'], ha=-newvis.phasecentre.ra.rad, dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad)[
                    ...]
            newvis.phasecentre = newphasecentre
        return newvis
    else:
        return vis

def predict_skycomponent_visibility_para(viss: list, sc: Skycomponent, mode="production"):
    '''
        并行版 predict_sky_component_visibility，结果不一致
    :param viss:
    :param sc:
    :return:
    '''
    # TODO 已修改
    l, m, n = skycoord_to_lmn(sc.direction, viss[0][1].phasecentre)
    for idx, vis in viss:
        phasor = simulate_point(vis.data['uvw'], l, m)
        if mode == "production":
            vis.data['vis'][0] += sc.flux[idx[2]//4, idx[5]] * phasor
        elif mode == "test":
            vis.data['vis'][0] += sc.flux[idx[0], idx[4]] * phasor
        elif mode == "test2":
            vis.data['vis'][0][idx[5]] += sc.flux[idx[0], idx[5]] * phasor

    return viss

def predict_skycoponent_visibility_para_modified(viss: list, sc: Skycomponent, mode="production"):
    '''
        改正了原本facet被多次加总的问题，最后结果一致
    :param viss:
    :param sc:
    :param mode:
    :return:
    '''
    l, m, n = skycoord_to_lmn(sc.direction, viss[0][1].phasecentre)
    fraction = 1.0 / PIECE
    for idx, vis in viss:
        phasor = simulate_point(vis.data['uvw'], l, m)
        if mode == "production":
            vis.data['vis'][0] += sc.flux[idx[2] // 4, idx[5]] * phasor * fraction
        elif mode == "test2":
            vis.data['vis'][0][idx[5]] += sc.flux[idx[0], idx[5]] * phasor * fraction

    return viss

# =============# =============# Timeslots =============# =============# =============
def solve_gaintable_para(vis: visibility_for_para, model: visibility_for_para=None, **kwargs):
    '''

    :param vis:
    :param model:
    :param phase_only:
    :param niter:
    :param tol:
    :param crosspol:
    :param kwargs:
    :return:
    '''
    pass

# =============# =============# Solve =============# =============# =============
def solve_from_X_para(x: numpy.ndarray, xwt: numpy.ndarray, phase_only=True, niter=30, tol=1e-8, crosspol=False, **kwargs)->gaintable_for_para:
    """

    :param vis:
    :param model:
    :param phase_only:
    :param niter:
    :param tol:
    :param crosspol:
    :param kwargs:
    :return:
    """
    pass

# =============# =============#  cor_subvis_flag  =============# =============# =============
def apply_gaintable(vis: visibility_for_para, gt: gaintable_for_para, inverse=False, **kwargs):
    """

    :param vis:
    :param gt:
    :param inverse:
    :param kwargs:
    :return:
    """
    pass
